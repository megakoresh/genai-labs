from download_and_extract import download_file
from dataclasses import dataclass
import argparse
import json
from gpt_utils import (
    calc_loss_loader_generator,
    plot_losses,
    separator,
    generate,
    token_ids_to_text,
    text_to_token_ids,
)
import os
from torch.utils.data import Dataset, DataLoader
import tiktoken
from typing import Callable, Iterable
import torch
from gpt_config import OpenAIModelConfigs
from functools import partial
from gpt2_model import (
    GPT2Model,
    load_weights_into_gpt_from_safetensors_params,
    train_generator_simple,
)
from gpt_download_pretrainged_weights import download_model_weights
import time
from tqdm import tqdm
import re


@dataclass
class CliArgs:
    dir: str
    url: str

    num_workers = 0
    batch_size = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and finetune a model for instruction task on given dataset"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="data/downloads",
        help="Folda where to save downloaded dataset",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Url from which to download instruction training dataset",
    )

    args = parser.parse_args()
    return CliArgs(url=args.url, dir=args.dir)


def format_input_alpaca(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(
        self,
        data: list[dict],
        tokenizer: tiktoken.Encoding,
        formatter: Callable[[dict], str],
    ):
        self.data = data
        self.encoded_texts = []
        for entry in data:  # 1
            instruction_plus_input = formatter(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_draft_1(
    batch: Iterable[list],
    device: torch.device,
    pad_token_id=50256,
):
    batch_max_length = max(len(item) + 1 for item in batch)  # 1
    inputs_lst = []

    for item in batch:  # 2
        new_item = item.copy()
        if len(new_item) < batch_max_length:
            padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        else:
            padded = new_item
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    inputs_tensor = torch.stack(inputs_lst).to(device)  # 4
    return inputs_tensor


def custom_collate_draft_2(
    batch: Iterable[list],
    device: torch.device,
    pad_token_id=50256,
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        if len(new_item) < batch_max_length:
            padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        else:
            padded = new_item
        inputs = torch.tensor(padded[:-1])  # 1
        targets = torch.tensor(padded[1:])  # 2
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def custom_collate_fn(
    batch: Iterable[list],
    device: torch.device,
    pad_token_id=50256,
    ignore_index=-100,  # default behavior of pytorch cross_entry function is to ignore targets labelled with -100
    allowed_max_length: int | None = None,
    instruction_length: int = -1,
):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        if len(new_item) < batch_max_length:
            padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        else:
            padded = new_item
        inputs = torch.tensor(padded[:-1])  # 2
        targets = torch.tensor(padded[1:])  # 3

        # replace all but the first pad token with ignore_index token
        # this means the model is not penalized for generating anything beyond the first eot token
        # in practice this means the model will not be trained to generate beyond the pad token, resulting chat-like behavior
        # where generation will be stopped when the first pad token is encountered
        mask = targets == pad_token_id  # 4
        indices = torch.nonzero(mask).squeeze()  # 4
        if indices.numel() > 1:  # 4
            targets[indices[1:]] = (
                ignore_index  # mask everything beyond the first pad token
            )

        # it is also common to mask out instruction part of the input so that model is not trained to memorize it
        # the same mechanism can be used for that
        if instruction_length > 0:
            targets[padded[:instruction_length]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]  # 5
            targets = targets[:allowed_max_length]  # 5

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


if __name__ == "__main__":
    args = parse_args()
    dataset_path = download_file(args.url, args.dir)
    tokenizer = tiktoken.get_encoding("gpt2")
    config = OpenAIModelConfigs.gpt2_med_255m
    file_name = (
        f"data/models/{re.sub(r'[^a-zA-Z0-9]+', '_', config.hf_repo_id) }-instruct.pth"
    )
    customized_collate_fn = partial(
        custom_collate_fn, device=config.device, allowed_max_length=1024
    )
    model = GPT2Model(config)

    with open(dataset_path, "r") as file:
        data = json.load(file)
    train_portion = int(len(data) * 0.85)  # 1
    test_portion = int(len(data) * 0.1)  # 2
    val_portion = len(data) - train_portion - test_portion  # 3

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    if os.path.exists(file_name):
        print(separator("Loading saved model"))
        model_state_dict = torch.load(file_name, map_location=config.device)
        model.load_state_dict(model_state_dict)
        model.to(config.device)
    else:
        print(separator("Training new instruction model"))
        model_dir, weights = download_model_weights(
            config.hf_repo_id, "data/models", "model.safetensors"
        )
        load_weights_into_gpt_from_safetensors_params(model, weights)
        model.to(config.device)
        print(separator("Dataset preview"))
        print("Number of entries:", len(data))
        print("Example entry:\n", data[50])
        print("Another example entry:\n", data[999])

        print("Changing input formats")

        model_input = format_input_alpaca(data[50])
        desired_response = f"\n\n### Response:\n{data[50]['output']}"
        print(model_input + desired_response)

        model_input = format_input_alpaca(data[999])
        desired_response = f"\n\n### Response:\n{data[999]['output']}"
        print(model_input + desired_response)

        print(separator("Dataset preparation"))

        print(
            "Tokenizer end of text token encoding",
            tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}),
        )
        print(separator("Testing collation"))
        inputs_1 = [0, 1, 2, 3, 4]
        inputs_2 = [5, 6]
        inputs_3 = [7, 8, 9]
        batch = (inputs_1, inputs_2, inputs_3)
        print("custom_collate_draft_1", custom_collate_draft_1(batch, config.device))

        inputs, targets = custom_collate_draft_2(batch, config.device)
        print("custom_collate_draft_2 inputs", inputs)
        print("custom_collate_draft_2 targets", targets)
        inputs, targets = customized_collate_fn(batch)
        print("final collate inputs:", inputs)
        print("final collate targets:", targets)

        print(separator("Data splitting"))

        torch.manual_seed(123)

        train_dataset = InstructionDataset(train_data, tokenizer, format_input_alpaca)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=customized_collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )

        val_dataset = InstructionDataset(val_data, tokenizer, format_input_alpaca)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            collate_fn=customized_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        )

        test_dataset = InstructionDataset(test_data, tokenizer, format_input_alpaca)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            collate_fn=customized_collate_fn,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        )
        input_text = format_input_alpaca(val_data[0])
        print("Input text:", input_text)
        input_tokens = text_to_token_ids(input_text, tokenizer)
        input_tokens = input_tokens.to(config.device)
        device = next(model.parameters()).device
        print("Selected device is", config.device)
        print("Model is on device", device)
        print("Input tokens are on device", input_tokens.device)
        token_ids = generate(
            model=model,
            idx=input_tokens,
            max_new_tokens=35,
            context_size=config.context_length,
            eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text) :].strip()
        print("Generated response", response_text)

        print(separator("Checking initial training and validatoin losses"))

        torch.manual_seed(123)

        with torch.no_grad():
            train_loss = calc_loss_loader_generator(
                train_loader, model, config.device, num_batches=5
            )
            val_loss = calc_loss_loader_generator(
                val_loader, model, config.device, num_batches=5
            )

        print("Training loss:", train_loss)
        print("Validation loss:", val_loss)

        print(separator(f"Fine-tuning model to instruction dataset"))

        start_time = time.time()
        torch.manual_seed(123)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
        num_epochs = 2

        train_losses, val_losses, tokens_seen = train_generator_simple(
            model,
            config,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=num_epochs,
            eval_freq=5,
            eval_iter=5,
            start_context=format_input_alpaca(val_data[0]),
            tokenizer=tokenizer,
        )

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")
        torch.save(model.state_dict(), file_name)
        print(f"Model saved as {file_name}")
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    for entry in test_data[:3]:  # 1
        input_text = format_input_alpaca(entry)
        token_ids = generate(  # 2
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(config.device),
            max_new_tokens=256,
            context_size=config.context_length,
            eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        response_text = (
            generated_text[len(input_text) :].replace("### Response:", "").strip()
        )
        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-------------------------------------")

        for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
            input_text = format_input_alpaca(entry)

            token_ids = generate(
                model=model,
                idx=text_to_token_ids(input_text, tokenizer).to(config.device),
                max_new_tokens=256,
                context_size=config.context_length,
                eos_id=50256,
            )
            generated_text = token_ids_to_text(token_ids, tokenizer)

            response_text = (
                generated_text[len(input_text) :].replace("### Response:", "").strip()
            )
            test_data[i]["model_response"] = response_text

        with open(f"{args.dir}/instruction-data-with-response.json", "w") as file:
            json.dump(test_data, file, indent=4)  # 1

        print(test_data[0])
