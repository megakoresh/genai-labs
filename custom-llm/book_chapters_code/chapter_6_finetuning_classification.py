import argparse
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tiktoken
import models.gpt2
from utils.gpt_download_pretrainged_weights import download_model_weights
from models.gpt2 import (
    GPT2Model,
    load_weights_into_gpt_from_safetensors_params,
)
import models.gpt2
from utils.gpt_utils import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
    separator,
    evaluate_model_classifier,
    calc_loss_batch_classifier,
    calc_loss_loader_classifier,
)
import time
import matplotlib.pyplot as plt
from torch import nn
import os


@dataclass
class CliArgs:
    name: str
    dir: str

    num_workers = 0
    batch_size = 8
    model_save_location = "data/models/review_classifier.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and finetune a model for classification task on given dataset"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data/prepared_datasets",
        help="Folda where the split datasets are located",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the dataset file without exitension and suffix e.g. for file SMSSpamCollection_test.csv it would be SMSSpamCollection. The given --dir must contain this dataset as 3 separate files with _test, _train and _validation suffixes",
    )

    args = parser.parse_args()
    return CliArgs(name=args.name, dir=args.dir)


class SpamDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        tokenizer: tiktoken.Encoding,
        max_length: int | None = None,
        pad_token_id=50256,
    ):
        self.data = pd.read_csv(csv_file)
        # 1
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 2
            self.encoded_texts = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        # 3
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index: int):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def get_dataset_file_paths(dir: str, name: str):
    return (
        f"{dir}/{name}_train.csv",
        f"{dir}/{name}_validation.csv",
        f"{dir}/{name}_test.csv",
    )


def calc_accuracy_loader(
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
            loss = calc_loss_batch_classifier(input_batch, target_batch, model)
            loss.backward()  # 5
            optimizer.step()  # 6
            examples_seen += input_batch.shape[0]  # 7
            global_step += 1

            # 8
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model_classifier(
                    model, train_loader, val_loader, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        # 9
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def plot_values(
    epochs_seen: torch.Tensor,
    examples_seen: torch.Tensor,
    train_values: list[float],
    val_values: list[float],
    label="loss",
):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 1
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # 2
    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)  # 3
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # 4
    if not os.path.exists("data/plots"):
        os.makedirs("data/plots")
    plt.savefig(f"data/plots/{label}-plot.pdf")
    plt.show()


def classify_review(
    text: str,
    model: GPT2Model,
    tokenizer: tiktoken.Encoding,
    device: torch.device,
    max_length: int,
    pad_token_id=50256,
):
    model.eval()

    input_ids = tokenizer.encode(text)  # 1
    supported_context_length = model.pos_emb.weight.shape[0]

    input_ids = input_ids[: min(max_length, supported_context_length)]  # 2

    input_ids += [pad_token_id] * (max_length - len(input_ids))  # 3

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # 4

    with torch.no_grad():  # 5
        logits = model(input_tensor)[:, -1, :]  # 6
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"  # 7


def adjust_model_for_classification(
    model: GPT2Model, num_classes: int, config: models.gpt2.GPTConfig
):
    # disable gradients on all layers so we only train the new finetuning layer
    for param in model.parameters():
        param.requires_grad = False
    model.out_head = torch.nn.Linear(
        in_features=config.emb_dim, out_features=num_classes, device=config.device
    )
    # also enable training on the last transformer block and output layer (optional but improves fine tuning performance)
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    args = parse_args()
    config = models.gpt2.OpenAIModelConfigs.gpt2_small_124m
    model_dir, weights = download_model_weights(
        "openai-community/gpt2", "data/models", "model.safetensors"
    )
    model = GPT2Model(config)
    load_weights_into_gpt_from_safetensors_params(model, weights)
    tokenizer = tiktoken.get_encoding("gpt2")
    train_ds_file_name, validation_ds_file_name, test_ds_file_name = (
        get_dataset_file_paths(args.dir, args.name)
    )
    train_dataset = SpamDataset(
        csv_file=train_ds_file_name,
        max_length=None,
        tokenizer=tokenizer,
    )
    model.eval()

    if os.path.exists(args.model_save_location):
        print(separator("Loading saved model"))
        adjust_model_for_classification(model, 2, config)
        model_state_dict = torch.load(
            args.model_save_location, map_location=config.device
        )
        model.load_state_dict(model_state_dict)
        model.to(config.device)
    else:
        print(separator("Training model"))
        val_dataset = SpamDataset(
            csv_file=validation_ds_file_name,
            max_length=train_dataset.max_length,
            tokenizer=tokenizer,
        )
        test_dataset = SpamDataset(
            csv_file=test_ds_file_name,
            max_length=train_dataset.max_length,
            tokenizer=tokenizer,
        )
        torch.manual_seed(123)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False,
        )

        for input_batch, target_batch in train_loader:
            print("Input batch dimensions:", input_batch.shape)
            print("Label batch dimensions", target_batch.shape)
            break

        print(f"{len(train_loader)} training batches")
        print(f"{len(val_loader)} validation batches")
        print(f"{len(test_loader)} test batches")

        input_prompt = "Every effort moves"
        token_ids = generate_text_simple(
            model,
            idx=text_to_token_ids(input_prompt, tokenizer),
            max_new_tokens=15,
            context_size=config.context_length,
        )
        print(token_ids_to_text(token_ids, tokenizer))

        print(separator("Text 2"))

        text_2 = (
            "Is the following text 'spam'? Answer with 'yes' or 'no':"
            " 'You are a winner you have been specially"
            " selected to receive $1000 cash or a $2000 award.'"
        )
        token_ids = generate_text_simple(
            model=model,
            idx=text_to_token_ids(text_2, tokenizer),
            max_new_tokens=23,
            context_size=config.context_length,
        )
        print(token_ids_to_text(token_ids, tokenizer))
        print(separator("Model architecture"))
        print(model)
        torch.manual_seed(123)
        adjust_model_for_classification(model, 2, config)
        print(separator("Dimensions preview"))
        inputs = tokenizer.encode("Do you have time")
        inputs = torch.tensor(inputs).unsqueeze(0)
        print("Inputs:", inputs)
        print("Inputs dimensions:", inputs.shape)  # 1
        with torch.no_grad():
            outputs = model(inputs)
        print("Outputs:\n", outputs)
        print("Outputs dimensions:", outputs.shape)
        print("Last output token:", outputs[:, -1, :])
        print(separator("Classification output preview"))
        probas = torch.softmax(outputs[:, -1, :], dim=-1)
        label = torch.argmax(probas)
        print("Class label:", label.item())
        logits = outputs[:, -1, :]
        label = torch.argmax(logits)
        print("Class label:", label.item())
        print(separator("calc_accuracy_loader test"))
        model.to(config.device)

        torch.manual_seed(123)
        train_accuracy = calc_accuracy_loader(
            train_loader, model, config.device, num_batches=10
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, config.device, num_batches=10
        )
        test_accuracy = calc_accuracy_loader(
            test_loader, model, config.device, num_batches=10
        )

        print(f"Training accuracy: {train_accuracy*100:.2f}%")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        print(f"Test accuracy: {test_accuracy*100:.2f}%")

        print(separator("calc_loss_loader test"))

        with torch.no_grad():  # 1
            train_loss = calc_loss_loader_classifier(train_loader, model, num_batches=5)
            val_loss = calc_loss_loader_classifier(val_loader, model, num_batches=5)
            test_loss = calc_loss_loader_classifier(test_loader, model, num_batches=5)
        print(f"Training loss: {train_loss:.3f}")
        print(f"Validation loss: {val_loss:.3f}")
        print(f"Test loss: {test_loss:.3f}")

        print(separator("Fine-tuning for classification"))

        start_time = time.time()
        torch.manual_seed(123)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
        num_epochs = 5

        train_losses, val_losses, train_accs, val_accs, examples_seen = (
            train_classifier_simple(
                model,
                train_loader,
                val_loader,
                optimizer,
                config.device,
                num_epochs=num_epochs,
                eval_freq=50,
                eval_iter=5,
            )
        )

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")
        print(separator("Plotting training loss"))

        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

        plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)
        print(separator("Plotting classification accuracies"))
        epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
        examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

        plot_values(
            epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy"
        )

        train_accuracy = calc_accuracy_loader(train_loader, model, config.device)
        val_accuracy = calc_accuracy_loader(val_loader, model, config.device)
        test_accuracy = calc_accuracy_loader(test_loader, model, config.device)

        print(separator("Final model performance"))

        print(f"Training accuracy: {train_accuracy*100:.2f}%")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        print(f"Test accuracy: {test_accuracy*100:.2f}%")

        torch.save(model.state_dict(), args.model_save_location)

    print(separator("Using the trained classifier"))

    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )

    print(
        text_1,
        classify_review(
            text_1, model, tokenizer, config.device, max_length=train_dataset.max_length
        ),
    )

    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print(
        text_2,
        classify_review(
            text_2, model, tokenizer, config.device, max_length=train_dataset.max_length
        ),
    )

    text_3 = "Dear sir or madame, have you heard about this new awesome deal? You can get 50$ in cashback for every 50$ purchase!"

    print(
        text_3,
        classify_review(
            text_2, model, tokenizer, config.device, max_length=train_dataset.max_length
        ),
    )
