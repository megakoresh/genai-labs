from typing import Callable, Iterable
import pandas as pd
import argparse
from dataclasses import dataclass
import os
import torch
from torch.utils.data import Dataset

from utils.gpt_utils import Encoding

RANDOM_STATE: int | None = 123


@dataclass
class CliArgs:
    input_file: str
    out_dir: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split and process source data into datasets for model training"
    )
    parser.add_argument(
        "--in-file",
        type=str,
        required=False,
        help="Tsv file to cload",
    )
    parser.add_argument(
        "--out-dir", type=str, required=False, default="data/prepared_datasets"
    )

    args = parser.parse_args()
    return CliArgs(input_file=args.in_file, out_dir=args.out_dir)


def load_datasource_file(file_path: str):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["Label", "Text"])
    return df


def create_balanced_dataset(df: pd.DataFrame):
    num_spam = df[df["Label"] == "spam"].shape[0]  # 1
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)  # 2
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])  # 3
    return balanced_df


def remap_labels(df: pd.DataFrame):
    df["Label"] = df["Label"].map({"ham": 0, "spam": 1})
    return df


def random_split(df: pd.DataFrame, train_fraction: float, validation_fraction: float):
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError(
            f"Sum of training fraction {train_fraction} and validation fraction {validation_fraction} is greater than 1.0, so no data is left for test set, please leave some data for testing"
        )

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    train_end = int(len(df) * train_fraction)
    validation_end = train_end + int(len(df) * validation_fraction)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def get_dataset_file_path(input_file: str, out_dir: str, suffix: str):
    source_filename = input_file.split(os.path.sep).pop()
    source_filename = source_filename.replace(".", "_")
    return f"{out_dir}/{source_filename}_{suffix}.csv"


def save_datasets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    input_file: str,
    out_dir: str,
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_dataset_file_path = get_dataset_file_path(input_file, out_dir, "train")
    validation_dataset_file_path = get_dataset_file_path(
        input_file, out_dir, "validation"
    )
    test_dataset_file_path = get_dataset_file_path(input_file, out_dir, "test")

    train_df.to_csv(train_dataset_file_path, index=False)
    validation_df.to_csv(validation_dataset_file_path, index=False)
    test_df.to_csv(test_dataset_file_path, index=False)

    return train_dataset_file_path, validation_dataset_file_path, test_dataset_file_path


def load_datasets(input_file: str, out_dir: str):
    train_dataset_file_path = get_dataset_file_path(input_file, out_dir, "train")
    validation_dataset_file_path = get_dataset_file_path(
        input_file, out_dir, "validation"
    )
    test_dataset_file_path = get_dataset_file_path(input_file, out_dir, "test")

    if (
        os.path.exists(train_dataset_file_path)
        and os.path.exists(validation_dataset_file_path)
        and os.path.exists(test_dataset_file_path)
    ):
        train_df = pd.read_csv(train_dataset_file_path)
        validation_df = pd.read_csv(validation_dataset_file_path)
        test_df = pd.read_csv(test_dataset_file_path)
        return train_df, validation_df, test_df

    return None, None, None


def main():
    args = parse_args()

    train_df, validation_df, test_df = load_datasets(args.input_file, args.out_dir)
    if train_df is None or validation_df is None or test_df is None:
        print(f"Did not find existing datasets, creating them")
        df = load_datasource_file(args.input_file)
        df = create_balanced_dataset(df)
        df = remap_labels(df)
        train_df, validation_df, test_df = random_split(df, 0.7, 0.1)
        (
            train_dataset_file_path,
            validation_dataset_file_path,
            test_dataset_file_path,
        ) = save_datasets(
            train_df, validation_df, test_df, args.input_file, args.out_dir
        )
        print(
            f"Saved {train_dataset_file_path}, {validation_dataset_file_path}, {test_dataset_file_path}"
        )


if __name__ == "__main__":
    main()


def format_input_alpaca(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


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

    inputs_tensor = torch.stack(inputs_lst)
    targets_tensor = torch.stack(targets_lst)
    return inputs_tensor, targets_tensor


class InstructionDataset(Dataset):
    def __init__(
        self,
        data: list[dict],
        tokenizer: Encoding,
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


class SpamDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        tokenizer: Encoding,
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
