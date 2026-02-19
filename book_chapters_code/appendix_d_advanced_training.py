import argparse
from dataclasses import dataclass
from functools import partial
import json
import math
import os
import random
import re
import time

import tiktoken
import torch
from tqdm import tqdm

from utils.downloads import download_file
from models.gpt2 import (
    GPT2Model,
    replace_linear_with_lora,
    train_generator_advanced,
)
from models.gpt2 import OpenAIModelConfigs
from utils.gpt_utils import (
    calc_loss_loader_generator,
    generate,
    separator,
    text_to_token_ids,
    token_ids_to_text,
)

from torch.utils.data import DataLoader

from utils.plot import plot_losses
from utils.datasets import (
    InstructionDataset,
    custom_collate_fn,
    format_input_alpaca,
)


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


if __name__ == "__main__":
    args = parse_args()
    rank = 32
    alpha = rank * 2
    dataset_path = download_file(args.url, args.dir)
    tokenizer = tiktoken.get_encoding("gpt2")
    config = OpenAIModelConfigs.gpt2_med_255m
    name = re.sub(r"[^a-zA-Z0-9]+", "_", config.hf_repo_id)
    file_name = f"data/models/{name}-instruct-lora-advanced.pth"
    customized_collate_fn = partial(
        custom_collate_fn, device=config.device, allowed_max_length=1024
    )
    model = GPT2Model(config)

    with open(dataset_path, "r") as file:
        data = json.load(file)
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

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

    if os.path.exists(file_name):
        print(separator("Loading saved model"))
        replace_linear_with_lora(model, rank, alpha)
        model_state_dict = torch.load(file_name, map_location=config.device)
        model.load_state_dict(model_state_dict)
        model.to(config.device)
    else:
        print(separator("Training model"))

        print("Training set length:", len(train_data))
        print("Validation set length:", len(val_data))
        print("Test set length:", len(test_data))

        print(separator("Freeze weights"))

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters before: {total_params:,}")

        for param in model.parameters():
            param.requires_grad = False
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters after: {total_params:,}")

        print(separator("Apply LoRA layers"))

        replace_linear_with_lora(model, rank=rank, alpha=alpha)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable LoRA parameters: {total_params:,}")
        model.to(config.device)

        print(separator("Evaluate model before fine-tuning"))

        torch.manual_seed(123)
        with torch.no_grad():
            train_accuracy = calc_loss_loader_generator(
                train_loader, model, config.device, num_batches=5
            )
            val_accuracy = calc_loss_loader_generator(
                val_loader, model, config.device, num_batches=5
            )
            test_accuracy = calc_loss_loader_generator(
                test_loader, model, config.device, num_batches=5
            )

        print(f"Training accuracy: {train_accuracy:.2f}%")
        print(f"Validation accuracy: {val_accuracy:.2f}%")
        print(f"Test accuracy: {test_accuracy:.2f}%")

        print(separator("Fine-tune model using LoRA layers"))

        start_time = time.time()

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
        num_epochs = 2
        train_losses, val_losses, examples_seen, lr_seen = train_generator_advanced(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=num_epochs,
            eval_freq=50,
            eval_iter=5,
            tokenizer=tokenizer,
            start_context=format_input_alpaca(
                val_data[int(random.random() * len(val_data))]
            ),
        )

        end_time = time.time()
        execution_time_minutes = (end_time - start_time) / 60
        print(f"Training completed in {execution_time_minutes:.2f} minutes.")

        torch.save(model.state_dict(), file_name)

        print(separator("Visualize losses"))

        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        examples_seen_tensor = torch.linspace(0, len(examples_seen), len(train_losses))

        plt = plot_losses(epochs_tensor, examples_seen_tensor, train_losses, val_losses)
        plt.savefig(f"data/plots/{name}-losses-lora-advanced-plot.pdf")
    print(separator("Evaluating test data"))
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

        with open(
            f"{args.dir}/instruction-data-with-response-{name}.json", "w"
        ) as file:
            json.dump(test_data, file, indent=4)  # 1

        print(test_data[0])
