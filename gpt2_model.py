import torch
from gpt_config import GPTConfig
from torch import nn
import numpy as np
from typing import Any
from torch.utils.data import DataLoader
import tiktoken
from gpt_utils import (
    calc_loss_batch_generator,
    calc_loss_batch_classifier,
    evaluate_model_classifier,
    evaluate_model_generator,
    generate,
    text_to_token_ids,
)


"""
Calculate multiple heads of attention at the same time using matrix maths 
instead of nested for loops
"""


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias=False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = (
            d_out // num_heads
        )  # 1 Reduces the projection dim to match the desired output dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(
            d_out, d_out
        )  # 2 Uses a Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)  # 3 Tensor shape: (b, num_tokens, d_out)
        queries = self.W_query(x)  # 3
        values = self.W_value(x)  # 3

        keys = keys.view(
            b, num_tokens, self.num_heads, self.head_dim
        )  # 4 We implicitly split the matrix by adding a num_heads dimension. Then we unroll the last dim: (b, num_tokens, d_out) -&gt; (b, num_tokens, num_heads, head_dim).
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(
            1, 2
        )  # 5 Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)  # 5
        values = values.transpose(1, 2)  # 5

        attn_scores = queries @ keys.transpose(
            2, 3
        )  # 6 Computes dot product for each head
        mask = self.mask
        assert isinstance(mask, torch.Tensor)
        mask_bool = mask.bool()[
            :num_tokens, :num_tokens
        ]  # 7 Masks truncated to the number of tokens

        attn_scores.masked_fill_(
            mask_bool, -torch.inf
        )  # 8 Uses the mask to fill attention scores

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(
            1, 2
        )  # 9 Tensor shape: (b, num_tokens, n_heads, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(
            context_vec
        )  # 10 Recombine results from all heads using linear layer
        return context_vec


# normalizes input to have mean of 0 and variance of 1
class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5  # prevents division by zero
        self.scale = nn.Parameter(
            torch.ones(emb_dim)
        )  # some "arbitrary" parameters to make the layer trainable
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        # does not apply Bessel's correction. Also this is default. In reality the correction does not matter for vectors with high dimensionality
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return (
            self.scale * norm_x + self.shift
        )  # I dont understand exactly why they are used like this, but some eggheads have figured it out so I trust them


# Gaussian Error Linear Unit activation function. Determines which positions in the output are "activated" (i.e. not set to zero)
# Has same shape as ReLU, but smoother and with a bend for lower values instead of just being flat at zero
# In practice it is found to offer better training performance than ReLU
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


"""
Standard processing layer in a NN that expands dimensionality and then contracts it using trainable linear layers
"""


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


"""
Transformer block is one layer of an LLM
It consists of attention layer and a feed forward layer, with normalization in between (but not after - normalization between blocks is done at model level)
It also applies dropout to prevent overdependence on individual outliers
It implements shortcut connections by adding original input to the output, which helps to combat vanishing gradient issue (when loss in earlier layers quickly becomes zero)
"""


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_attn_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor):
        # 1
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 2

        shortcut = x  # 3
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 4
        return x


"""
Basic model representing OpenAI GPT-2 model architecture
"""


class GPT2Model(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(
            cfg.vocab_size, cfg.emb_dim
        )  # token embeddings with vocabulary
        self.pos_emb = nn.Embedding(
            cfg.context_length, cfg.emb_dim
        )  # word position embeddings
        self.drop_emb = nn.Dropout(
            cfg.drop_rate
        )  # dropout layer to prevent overfitting

        # model layers
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        # pre-output normalization
        self.final_norm = LayerNorm(cfg.emb_dim)
        # replacable final output layer (not normalized)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(
            x
        )  # logits are unscaled output (i.e. non-normalized) after expanding model output to vocabulary size
        return logits


def assign(
    left: np.typing.NDArray[Any] | torch.Tensor,
    right: np.typing.NDArray[Any] | torch.Tensor,
):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, " f"Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


def assign_transformer_block_openai(block: TransformerBlock, params: dict[str, Any]):
    q_w, k_w, v_w = np.split(  # initialization weights for K, Q and V
        (params["attn"]["c_attn"])["w"], 3, axis=-1
    )
    block.att.W_query.weight = assign(block.att.W_query.weight, q_w.T)
    block.att.W_key.weight = assign(block.att.W_key.weight, k_w.T)
    block.att.W_value.weight = assign(block.att.W_value.weight, v_w.T)

    q_b, k_b, v_b = np.split((params["attn"]["c_attn"])["b"], 3, axis=-1)
    block.att.W_query.bias = assign(block.att.W_query.bias, q_b)
    block.att.W_key.bias = assign(block.att.W_key.bias, k_b)
    block.att.W_value.bias = assign(block.att.W_value.bias, v_b)

    block.att.out_proj.weight = assign(
        block.att.out_proj.weight, params["attn"]["c_proj"]["w"].T
    )
    block.att.out_proj.bias = assign(
        block.att.out_proj.bias, params["attn"]["c_proj"]["b"]
    )

    expand_layer = block.ff.layers[0]
    assert isinstance(expand_layer, nn.Module)
    assert isinstance(expand_layer.weight, torch.Tensor)
    assert isinstance(expand_layer.bias, torch.Tensor)
    contract_layer = block.ff.layers[2]
    assert isinstance(contract_layer, nn.Module)
    assert isinstance(contract_layer.weight, torch.Tensor)
    assert isinstance(contract_layer.bias, torch.Tensor)

    block.ff.layers[0].weight = assign(
        expand_layer.weight, params["mlp"]["c_fc"]["w"].T
    )
    block.ff.layers[0].bias = assign(expand_layer.bias, params["mlp"]["c_fc"]["b"])
    block.ff.layers[2].weight = assign(
        contract_layer.weight, params["mlp"]["c_proj"]["w"].T
    )
    block.ff.layers[2].bias = assign(contract_layer.bias, params["mlp"]["c_proj"]["b"])

    block.norm1.scale = assign(block.norm1.scale, params["ln_1"]["g"])
    block.norm1.shift = assign(block.norm1.shift, params["ln_1"]["b"])
    block.norm2.scale = assign(block.norm2.scale, params["ln_2"]["g"])
    block.norm2.shift = assign(block.norm2.shift, params["ln_2"]["b"])
    return block


def load_weights_into_gpt_from_openai(gpt: GPT2Model, params: dict[str, Any]):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(gpt.trf_blocks)):
        block = gpt.trf_blocks[b]
        assert isinstance(block, TransformerBlock)
        block_params = params["blocks"][b]
        gpt.trf_blocks[b] = assign_transformer_block_openai(block, block_params)

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(
        gpt.out_head.weight, params["wte"]
    )  # original model resused the same weights as for input layer, but the book recommends not to do this, as it negatively impacts model performance


def assign_transformer_block_safetensors(
    block: TransformerBlock, params: dict, b_idx: int
):
    q_w, k_w, v_w = torch.chunk(params[f"h.{b_idx}.attn.c_attn.weight"], 3, dim=-1)
    block.att.W_query.weight = assign(block.att.W_query.weight, q_w.T)
    block.att.W_key.weight = assign(block.att.W_key.weight, k_w.T)
    block.att.W_value.weight = assign(block.att.W_value.weight, v_w.T)

    q_b, k_b, v_b = torch.chunk(params[f"h.{b_idx}.attn.c_attn.bias"], 3, dim=-1)
    block.att.W_query.bias = assign(block.att.W_query.bias, q_b)
    block.att.W_key.bias = assign(block.att.W_key.bias, k_b)
    block.att.W_value.bias = assign(block.att.W_value.bias, v_b)

    block.att.out_proj.weight = assign(
        block.att.out_proj.weight, params[f"h.{b_idx}.attn.c_proj.weight"].T
    )
    block.att.out_proj.bias = assign(
        block.att.out_proj.bias, params[f"h.{b_idx}.attn.c_proj.bias"]
    )

    expand_layer = block.ff.layers[0]
    assert isinstance(expand_layer, nn.Module)
    assert isinstance(expand_layer.weight, torch.Tensor)
    assert isinstance(expand_layer.bias, torch.Tensor)
    contract_layer = block.ff.layers[2]
    assert isinstance(contract_layer, nn.Module)
    assert isinstance(contract_layer.weight, torch.Tensor)
    assert isinstance(contract_layer.bias, torch.Tensor)

    block.ff.layers[0].weight = assign(
        expand_layer.weight, params[f"h.{b_idx}.mlp.c_fc.weight"].T
    )
    block.ff.layers[0].bias = assign(
        expand_layer.bias, params[f"h.{b_idx}.mlp.c_fc.bias"]
    )
    block.ff.layers[2].weight = assign(
        contract_layer.weight, params[f"h.{b_idx}.mlp.c_proj.weight"].T
    )
    block.ff.layers[2].bias = assign(
        contract_layer.bias, params[f"h.{b_idx}.mlp.c_proj.bias"]
    )

    block.norm1.scale = assign(block.norm1.scale, params[f"h.{b_idx}.ln_1.weight"])
    block.norm1.shift = assign(block.norm1.shift, params[f"h.{b_idx}.ln_1.bias"])
    block.norm2.scale = assign(block.norm2.scale, params[f"h.{b_idx}.ln_2.weight"])
    block.norm2.shift = assign(block.norm2.shift, params[f"h.{b_idx}.ln_2.bias"])
    return block


def load_weights_into_gpt_from_safetensors_params(gpt: GPT2Model, params: dict):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe.weight"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte.weight"])
    for b in range(len(gpt.trf_blocks)):
        block = gpt.trf_blocks[b]
        assert isinstance(block, TransformerBlock)
        gpt.trf_blocks[b] = assign_transformer_block_safetensors(block, params, b)

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["ln_f.bias"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte.weight"])


def classification_accuracy_loader(
    data_loader: DataLoader,
    model: GPT2Model,
    device: torch.device,
    num_batches: int | None = None,
):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # 1
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()

        else:
            break
    return correct_predictions / num_examples


def train_classifier_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []  # 1
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):  # 2
        model.train()  # 3

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 4
            loss = calc_loss_batch_classifier(input_batch, target_batch, model, device)
            loss.backward()  # 5
            optimizer.step()  # 6
            examples_seen += input_batch.shape[0]  # 7
            global_step += 1

            # 8
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_classifier(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        # 9
        train_accuracy = classification_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = classification_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def train_generator_simple(
    model: GPT2Model,
    config: GPTConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer: tiktoken.Encoding,
):
    train_losses, val_losses, track_tokens_seen = [], [], []  # training monitors
    tokens_seen, global_step = 0, -1
    # according to book, most of the production models are trained a few times on huge corpi of data, rather than many times on small corpus like here. This is done to prevent overfitting
    for epoch in range(num_epochs):
        model.train()
        for i, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()  # reset gradients from prev. epoch
            loss = calc_loss_batch_generator(input_batch, target_batch, model, device)
            loss.backward()  # update gradients via backpropagation
            optimizer.step()  # update model weights based on gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # i struggled to understand why the batch is strangely aligned with the iter_freq, but its because of the data split. When split at 0.90, that means train_loader has 9 batches in total and val_loader has 1
            # it seems that its just a conincidence that total number of batches is 10 in "the verdict"
            # print("batch no", i)

            # check model performance every eval_freq steps
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_generator(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        sample = generate(
            model,
            text_to_token_ids(start_context, tokenizer),
            50,
            config.context_length,
            1.5,
            15,
            50256,
        )  # 7
        print("Sample:", sample)
    return train_losses, val_losses, track_tokens_seen
