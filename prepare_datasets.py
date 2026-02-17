import pandas as pd
import argparse
from dataclasses import dataclass
import os
import tiktoken
import torch
from torch.utils.data import Dataset

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
