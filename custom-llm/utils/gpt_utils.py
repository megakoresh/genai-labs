import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import AbstractSet, Collection, Literal, Protocol, Sequence


class Encoding(Protocol):
    def encode(self, text: str, *, allowed_special: AbstractSet[str] | Literal["all"] = set(), disallowed_special: Collection[str] | Literal["all"] = "all") -> list[int]:  # type: ignore
        pass

    def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:  # type: ignore
        pass


def generate_text_simple(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    disable_grad=False,
):
    for _ in range(max_new_tokens):
        idx_cond = idx[
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
        idx = torch.cat(
            (idx, idx_next), dim=1
        )  # 6 concatenate the predicted token with the input

    return idx


# tokenize text
def text_to_token_ids(text: str, tokenizer: Encoding):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 1 restore batch dimension
    return encoded_tensor


# decode token ids
def token_ids_to_text(token_ids: torch.Tensor, tokenizer: Encoding):
    flat = token_ids.squeeze(0)  # 2 remove batch dimension
    return tokenizer.decode(flat.tolist())


def separator(title: str):
    length = 48
    sep = "="
    assert len(title) < length
    n_sep = int(length - (len(title)) / 2)
    return (sep * n_sep) + title + (sep * n_sep)


def calc_loss_batch_classifier(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
):
    logits = model(input_batch)[:, -1, :]  # 1
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_batch_generator(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
):
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader_classifier(
    data_loader: DataLoader,
    model: nn.Module,
    num_batches: int | None = None,
):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:  # 1
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch_classifier(input_batch, target_batch, model)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def calc_loss_loader_generator(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:  # 1
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch_generator(input_batch, target_batch, model)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    eval_iter: int,
):
    model.eval()  # disable backpropagation for evaluation run
    with torch.no_grad():  # disable gradients for evaluation run
        train_loss = calc_loss_loader_classifier(
            train_loader, model, num_batches=eval_iter
        )
        val_loss = calc_loss_loader_classifier(val_loader, model, num_batches=eval_iter)
    model.train()  # set back to training mode
    return train_loss, val_loss


def evaluate_model_generator(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
):
    model.eval()  # disable backpropagation for evaluation run
    with torch.no_grad():  # disable gradients for evaluation run
        train_loss = calc_loss_loader_generator(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader_generator(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()  # set back to training mode
    return train_loss, val_loss


def generate(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature=0.0,
    top_k: int | None = None,
    eos_id: int | None = None,
):
    device = next(model.parameters()).device
    idx = idx.to(device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        # evaluate according to topk sampling (top k largest logits)
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )
        # apply temperature scaling if >0
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:  # otherwise use argmax to select biggest one
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:  # stop generating if eos is encountered
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
