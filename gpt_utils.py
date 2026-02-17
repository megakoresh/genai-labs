import torch
from torch import nn
import tiktoken
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List


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


def calc_loss_batch_classifier(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # 1
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_batch_generator(
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


def calc_loss_loader_classifier(
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
            loss = calc_loss_batch_classifier(input_batch, target_batch, model, device)
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
            loss = calc_loss_batch_generator(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
):
    model.eval()  # disable backpropagation for evaluation run
    with torch.no_grad():  # disable gradients for evaluation run
        train_loss = calc_loss_loader_classifier(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader_classifier(
            val_loader, model, device, num_batches=eval_iter
        )
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
