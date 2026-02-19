import json
import os
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from dataclasses import dataclass
import argparse
import re


@dataclass
class _CliArgs:
    repo_id: str
    base_dir: str
    tensors_file: str


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Download model safetensors file from huggingface"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/models",
        required=False,
        help="Base directory where the model files will be saved",
    )

    parser.add_argument(
        "--repo-id", type=str, required=True, help="Repo ID of the model"
    )
    parser.add_argument(
        "--tensors-file",
        type=str,
        required=False,
        help="safetensors file in the repo",
        default="model.safetensors.index.json",
    )
    args = parser.parse_args()
    return _CliArgs(
        repo_id=args.repo_id, base_dir=args.base_dir, tensors_file=args.tensors_file
    )


def download_model_weights(repo_id: str, base_dir: str, tensors_file_or_index: str):
    base_dir = f"{base_dir}/{re.sub(r"[^a-zA-Z0-9]", "_", repo_id)}"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=base_dir)
    index_path = os.path.join(repo_dir, tensors_file_or_index)
    weights_dict = {}
    if os.path.exists(index_path) and index_path.endswith("index.json"):
        with open(index_path, "r") as index_f:
            index = json.load(index_f)
        weights_map = index["weight_map"]
        assert isinstance(weights_map, dict)
        for filename in set(weights_map.values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)
        return index_path, weights_dict
    else:
        shard_path = os.path.join(repo_dir, tensors_file_or_index)
        print(f"No index file found, loading {shard_path} as a single shard")
        shard = load_file(shard_path)
        weights_dict.update(shard)
        return shard_path, weights_dict


if __name__ == "__main__":
    args = _parse_args()
    safetensors_path, weights = download_model_weights(
        args.repo_id, args.base_dir, args.tensors_file
    )
    print(
        f"Downloaded model keys from {args.repo_id} to {safetensors_path}. Available parameters:"
    )
    for key in weights.keys():
        print(" ", key)
