from dataclasses import dataclass
from typing import List, Any
from torch import nn
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import urllib.request
import importlib.util
import sys
import numpy as np

INSPECT = os.environ.get("INSPECT") in ["yes", "true", "1"]
EOT_TOKEN = "<|endoftext|>"
UNK_TOKEN = "<|unk|>"
SPLIT_REGEX = r'([,.:;?_!"()\']|--|\s)'


@dataclass
class GPTConfig:
    context_length: int
    vocab_size: int
    emb_dim: int
    n_attn_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool

    _device: torch.device | None = None

    @property
    def device(self):
        if self._device == None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self._device


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


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        # 1
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(
            x
        )  # logits are unscaled output after expanding model output to vocabulary size
        return logits


# runs the model to generate up to max_new_tokens
def generate_text_simple(
    model: nn.Module,
    in_idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    disable_grad=False,
):
    for _ in range(max_new_tokens):
        idx_cond = in_idx[
            :, -context_size:
        ]  # 2 truncate input to context size (from beginning - I wonder what would happen if it was truncated from end...)
        if disable_grad:
            with torch.no_grad():
                logits = model(idx_cond)
        else:
            logits = model(idx_cond)

        logits = logits[
            :, -1, :
        ]  # 3 we only care about the last item (i.e. the predicted token)
        probas = torch.softmax(
            logits, dim=-1
        )  # 4 probability distribution across the vocabulary dimension
        idx_next = torch.argmax(
            probas, dim=-1, keepdim=True
        )  # 5 select one with greatest probability
        in_idx = torch.cat(
            (in_idx, idx_next), dim=1
        )  # 6 concatenate the predicted token with the input

    return in_idx


# tokenize text
def text_to_token_ids(text: str, tokenizer: tiktoken.Encoding):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 1 restore batch dimension
    return encoded_tensor


# decode token ids
def token_ids_to_text(token_ids: torch.Tensor, tokenizer: tiktoken.Encoding):
    flat = token_ids.squeeze(0)  # 2 remove batch dimension
    return tokenizer.decode(flat.tolist())


def separator(title: str):
    length = 48
    sep = "="
    assert len(title) < length
    n_sep = int(length - (len(title)) / 2)
    return (sep * n_sep) + title + (sep * n_sep)


class GPTDatasetV1(Dataset):
    def __init__(
        self, txt: str, tokenizer: tiktoken.Encoding, input_window: int, stride: int
    ) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={EOT_TOKEN})

        for i in range(0, len(token_ids) - input_window, stride):
            input_chunk = token_ids[i : i + input_window]
            target_chunk = token_ids[i + 1 : i + input_window + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    tokenizer_name: str,
    batch_size: int,
    input_window: int,
    stride: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
):
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    dataset = GPTDatasetV1(txt, tokenizer, input_window, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)  # if upper limit not defined, process all
    else:
        num_batches = min(
            num_batches, len(data_loader)
        )  # else if limit is less than length of loader, set limit to length of loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()  # accumulate losses
        else:
            break
    return total_loss / num_batches  # total loss is the average of losses in each batch


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
):
    model.eval()  # disable backpropagation for evaluation run
    with torch.no_grad():  # disable gradients for evaluation run
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # set back to training mode
    return train_loss, val_loss


def generate_and_print_sample(
    model: nn.Module,
    tokenizer: tiktoken.Encoding,
    device: torch.device,
    start_context: str,
):
    model.eval()
    assert isinstance(model.pos_emb, nn.Embedding)
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, in_idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # print model output as text for preview
    model.train()


def train_model_simple(
    model: nn.Module,
    train_loader: DataLoader[GPTDatasetV1],
    val_loader: DataLoader[GPTDatasetV1],
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
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # update gradients via backpropagation
            optimizer.step()  # update model weights based on gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # i struggled to understand why the batch is strangely aligned with the iter_freq, but its because of the data split. When split at 0.90, that means train_loader has 9 batches in total and val_loader has 1
            # it seems that its just a conincidence that total number of batches is 10 in "the verdict"
            # print("batch no", i)

            # check model performance every eval_freq steps
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
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

        generate_and_print_sample(model, tokenizer, device, start_context)  # 7
    return train_losses, val_losses, track_tokens_seen


def plot_losses(
    epochs_seen: torch.Tensor,
    tokens_seen: List[float],
    train_losses: List[float],
    val_losses: List[float],
):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()  # second x axis for for tokens seen
    ax2.plot(
        tokens_seen, train_losses, alpha=0
    )  # invisible plot for aligning ticks (wat?)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()


def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature=0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if (
            top_k is not None
        ):  # evaluate according to topk sampling (top k largest logits)
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )
        if temperature > 0.0:  # apply temperature scaling if >0
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:  # otherwise use argmax to select biggest one
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:  # stop generating if eos is encountered
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def chapter_5_1():
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    model_cfg = GPTConfig(
        context_length=256,
        vocab_size=50257,
        emb_dim=768,
        n_attn_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False,
    )
    model = GPTModel(model_cfg)

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 1
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval()  # 1 disables dropout for inference
    out = generate_text_simple(
        model=model,
        in_idx=encoded_tensor,
        max_new_tokens=6,
        context_size=model_cfg.context_length,
    )
    print("Output:", out)
    print("Output length:", len(out[0]))
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)

    print(separator("With loss calculation"))

    inputs = torch.tensor(
        [[16833, 3626, 6100], [40, 1107, 588]]  # ["every effort moves",
    )  #  "I really like"]
    targets = torch.tensor(
        [[3626, 6100, 345], [1107, 588, 11311]]  # [" effort moves you",
    )  #  " really like chocolate"]
    with torch.no_grad():  # 1 not training yet, disable gradients
        logits = model(inputs)
    probas = torch.softmax(logits, dim=-1)  # 2 calculate probabilities over vocabulary
    print(
        "Probas shape:", probas.shape
    )  # each batch has 3 tokens and each token is a prob. distribution over vocabulary
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)  # predicted tokens
    print("Token IDs:\n", token_ids)
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(
        f"Outputs batch 1:" f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}"
    )
    """
    Here is unintuitive part:
    The goal of training is to make sure the maximum probabilities returned by the model are at the indices corresponding to the target tokens in the vocabulary
    We do not care what the probabilities actually are, only that the maximum is corresponding to the target token
    Here we get the target probabilities from the predited output (which is gibberish)
    This means the probabilities for the target token ids are lower than what they should be
    We define the loss function as the average of the natural logarithm of these outputs
    The natural logarithm is zero when the outputs are equal to 1 (i.e. when the model prediction is perfect)
    But we should never make it perfect, otherwise it overfits! We only need the maximum value of the 50257-length distribution to match the target token
    I do not understand yet how is this issue solved
    """
    batch_idx = 0
    target_probas_1 = probas[batch_idx, [0, 1, 2], targets[batch_idx]]
    print("Text 1 target probas:", target_probas_1)

    batch_idx = 1
    target_probas_2 = probas[batch_idx, [0, 1, 2], targets[batch_idx]]
    print("Text 2 target probas:", target_probas_2)
    # cross entry loss:
    # natural log is computationally easier to work with
    # very smol values -> -inf; 0 when x=1
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print("Log probas", log_probas)
    avg_log_probas = torch.mean(log_probas)
    print("Avg log probas", avg_log_probas)
    neg_avg_log_probas = avg_log_probas * -1
    print(
        "Negative avg log probas", neg_avg_log_probas
    )  # <- this is what we want to get closer to zero

    # the same as above, but from pytorch
    logits_flat = logits.flatten(
        0, 1
    )  # flatten batch dimension (concatenate all words/tokens from all batches)
    targets_flat = (
        targets.flatten()
    )  # targets are token ids, so they dont have dictionary dimension
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print("Cross entropy loss", loss)

    """
    Quote:
    Perplexity is often considered more interpretable than the raw loss value because it signifies the effective vocabulary size about which the model is uncertain at each step. 
    In the given example, this would translate to the model being unsure about which among 48,725 tokens in the vocabulary to generate as the next token.
    """
    perplexity = torch.exp(loss)
    print("Perplexity:", perplexity)

    print(separator("With training dataset"))

    with open("data/samples/the-verdict.txt", encoding="utf-8") as sample:
        text_data = sample.read()
        total_chars = len(text_data)
        total_tokens = len(tokenizer.encode(text_data))
        print("Characters:", total_chars)
        print("Tokens:", total_tokens)

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    train_loader = create_dataloader_v1(
        txt=train_data,
        tokenizer_name="gpt2",
        batch_size=2,
        input_window=model_cfg.context_length,
        stride=model_cfg.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        txt=val_data,
        tokenizer_name="gpt2",
        batch_size=2,
        input_window=model_cfg.context_length,
        stride=model_cfg.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    model.to(model_cfg.device)  # use gpu if available
    with torch.no_grad():  # disable gradients for loss function preview
        train_loss = calc_loss_loader(
            train_loader, model, model_cfg.device
        )  # good idea to pass device everywhere so that everything can load data to same device
        val_loss = calc_loss_loader(val_loader, model, model_cfg.device)
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)


"""
TODO: 
Interested readers can also use the supplementary code for this book to prepare a larger-scale dataset consisting 
of more than 60,000 public domain books from Project Gutenberg and train an LLM on these (see appendix D for details).
"""


def chapter_5_2():
    torch.manual_seed(123)
    model_cfg = GPTConfig(
        context_length=256,
        vocab_size=50257,
        emb_dim=768,
        n_attn_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False,
    )
    model = GPTModel(model_cfg)
    model.to(model_cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)  # 1
    tokenizer = tiktoken.get_encoding("gpt2")
    num_epochs = 10
    train_ratio = 0.90

    with open("data/samples/the-verdict.txt", encoding="utf-8") as sample:
        text_data = sample.read()
        total_chars = len(text_data)
        total_tokens = len(tokenizer.encode(text_data))
        print("Characters:", total_chars)
        print("Tokens:", total_tokens)

    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    train_loader = create_dataloader_v1(
        txt=train_data,
        tokenizer_name="gpt2",
        batch_size=2,
        input_window=model_cfg.context_length,
        stride=model_cfg.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        txt=val_data,
        tokenizer_name="gpt2",
        batch_size=2,
        input_window=model_cfg.context_length,
        stride=model_cfg.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print("Train loader length:", len(train_loader))
    print("Validation loader length:", len(val_loader))

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        model_cfg.device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    # epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    # plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    model.to("cpu")
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
        model=model,
        in_idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=model_cfg.context_length,
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


def chapter_5_3():
    torch.manual_seed(123)
    model_cfg = GPTConfig(
        context_length=256,
        vocab_size=50257,
        emb_dim=768,
        n_attn_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False,
    )
    model = GPTModel(model_cfg)
    model.to(model_cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)  # 1
    tokenizer = tiktoken.get_encoding("gpt2")
    num_epochs = 10
    train_ratio = 0.90

    with open("data/samples/the-verdict.txt", encoding="utf-8") as sample:
        text_data = sample.read()
        total_chars = len(text_data)
        total_tokens = len(tokenizer.encode(text_data))
        print("Characters:", total_chars)
        print("Tokens:", total_tokens)

    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    train_loader = create_dataloader_v1(
        txt=train_data,
        tokenizer_name="gpt2",
        batch_size=2,
        input_window=model_cfg.context_length,
        stride=model_cfg.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        txt=val_data,
        tokenizer_name="gpt2",
        batch_size=2,
        input_window=model_cfg.context_length,
        stride=model_cfg.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print("Train loader length:", len(train_loader))
    print("Validation loader length:", len(val_loader))

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        model_cfg.device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )
    model.to("cpu")
    model.eval()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=model_cfg.context_length,
        top_k=25,
        temperature=1.4,
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


def chapter_5_4():
    fname = "data/models/gpt_model_chapter_5_4.pth"
    model_cfg = GPTConfig(
        context_length=256,
        vocab_size=50257,
        emb_dim=768,
        n_attn_heads=12,
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False,
    )
    if os.path.isfile(fname):
        model = GPTModel(model_cfg)
        model.to(model_cfg.device)
        checkpoint = torch.load(fname, map_location=model_cfg.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.train()
        num_epochs = 1
    else:
        model = GPTModel(model_cfg)
        model.to(model_cfg.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
        num_epochs = 10

    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")
    train_ratio = 0.90

    with open("data/samples/the-verdict.txt", encoding="utf-8") as sample:
        text_data = sample.read()
        total_chars = len(text_data)
        total_tokens = len(tokenizer.encode(text_data))
        print("Characters:", total_chars)
        print("Tokens:", total_tokens)

    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    train_loader = create_dataloader_v1(
        txt=train_data,
        tokenizer_name="gpt2",
        batch_size=2,
        input_window=model_cfg.context_length,
        stride=model_cfg.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        txt=val_data,
        tokenizer_name="gpt2",
        batch_size=2,
        input_window=model_cfg.context_length,
        stride=model_cfg.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print("Train loader length:", len(train_loader))
    print("Validation loader length:", len(val_loader))

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        model_cfg.device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            # saving optimizer is recommended if we are planning to continue training later
            "optimizer_state_dict": optimizer.state_dict(),
        },
        fname,
    )


def assign(
    left: np.typing.NDArray[Any] | torch.Tensor,
    right: np.typing.NDArray[Any] | torch.Tensor,
):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, " f"Right: {right.shape}"
        )
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt: GPTModel, params: dict[str, Any]):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(  # initialization weights for K, Q and V
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        block = gpt.trf_blocks[b]
        assert isinstance(block, TransformerBlock)
        block.att.W_query.weight = assign(block.att.W_query.weight, q_w.T)
        block.att.W_key.weight = assign(block.att.W_key.weight, k_w.T)
        block.att.W_value.weight = assign(block.att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        block.att.W_query.bias = assign(block.att.W_query.bias, q_b)
        block.att.W_key.bias = assign(block.att.W_key.bias, k_b)
        block.att.W_value.bias = assign(block.att.W_value.bias, v_b)

        block.att.out_proj.weight = assign(
            block.att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        block.att.out_proj.bias = assign(
            block.att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"]
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
            expand_layer.weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        block.ff.layers[0].bias = assign(
            expand_layer.bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        block.ff.layers[2].weight = assign(
            contract_layer.weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        block.ff.layers[2].bias = assign(
            contract_layer.bias, params["blocks"][b]["mlp"]["c_proj"]["b"]
        )

        block.norm1.scale = assign(block.norm1.scale, params["blocks"][b]["ln_1"]["g"])
        block.norm1.shift = assign(block.norm1.shift, params["blocks"][b]["ln_1"]["b"])
        block.norm2.scale = assign(block.norm2.scale, params["blocks"][b]["ln_2"]["g"])
        block.norm2.shift = assign(block.norm2.shift, params["blocks"][b]["ln_2"]["b"])

        gpt.trf_blocks[b] = block

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(
        gpt.out_head.weight, params["wte"]
    )  # original model resused the same weights as for input layer, but the book recommends not to do this, as it negatively impacts model performance


def chapter_5_5():
    dl_script = "./data/scripts/gpt_download.py"
    if not os.path.isfile(dl_script):
        url = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch05/"
            "01_main-chapter-code/gpt_download.py"
        )
        urllib.request.urlretrieve(url, dl_script)
    spec = importlib.util.spec_from_file_location("gpt_download", dl_script)
    assert spec != None
    gpt_download = importlib.util.module_from_spec(spec)
    sys.modules["gpt_download"] = gpt_download
    assert spec.loader != None
    spec.loader.exec_module(gpt_download)
    settings, params = gpt_download.download_and_load_gpt2(
        model_size="124M", models_dir="data/models"
    )
    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())
    print(params["wte"])
    print("Token embedding weight tensor dimensions:", params["wte"].shape)

    GPT2_SMALL_124M = "gpt2-small (124M)"
    GPT2_MEDIUM_255M = "gpt2-medium (355M)"
    GPT2_LARGE_774M = "gpt2-large (774M)"
    GPT_XLARGE_1558M = "gpt2-xl (1558M)"

    model_configs = {
        GPT2_SMALL_124M: GPTConfig(
            emb_dim=768,
            n_layers=12,
            n_attn_heads=12,
            context_length=1024,
            vocab_size=50257,
            drop_rate=0.1,
            qkv_bias=True,
        ),
        GPT2_MEDIUM_255M: GPTConfig(
            emb_dim=1024,
            n_layers=24,
            n_attn_heads=16,
            context_length=1024,
            vocab_size=50257,
            drop_rate=0.1,
            qkv_bias=True,
        ),
        GPT2_LARGE_774M: GPTConfig(
            emb_dim=1280,
            n_layers=36,
            n_attn_heads=20,
            context_length=1024,
            vocab_size=50257,
            drop_rate=0.1,
            qkv_bias=True,
        ),
        GPT_XLARGE_1558M: GPTConfig(
            emb_dim=1600,
            n_layers=48,
            n_attn_heads=25,
            context_length=1024,
            vocab_size=50257,
            drop_rate=0.1,
            qkv_bias=True,
        ),
    }
    config = model_configs[GPT2_SMALL_124M]
    gpt = GPTModel(config)
    gpt.eval()
    load_weights_into_gpt(gpt, params)
    gpt.to(config.device)
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("There is no pancake mix in there", tokenizer).to(
            config.device
        ),
        max_new_tokens=25,
        context_size=config.context_length,
        top_k=50,
        temperature=1.5,
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    with open("data/samples/the-verdict.txt", encoding="utf-8") as sample:
        text_data = sample.read()
        total_chars = len(text_data)
        total_tokens = len(tokenizer.encode(text_data))
        print("Characters:", total_chars)
        print("Tokens:", total_tokens)

    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    train_loader = create_dataloader_v1(
        txt=train_data,
        tokenizer_name="gpt2",
        batch_size=2,
        input_window=config.context_length,
        stride=config.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        txt=val_data,
        tokenizer_name="gpt2",
        batch_size=2,
        input_window=config.context_length,
        stride=config.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    print("Train loader length", len(train_loader))
    print("Validation loader length", len(val_loader))

    train_loss = calc_loss_loader(train_loader, gpt, config.device)
    val_loss = calc_loss_loader(val_loader, gpt, config.device)
    print(
        f"Train loss on 'The Verdict' {train_loss:.3f}, "
        f"Val loss on 'The Verdict' {val_loss:.3f}"
    )


def main():
    pass


if __name__ == "__main__":
    chapter_5_5()
