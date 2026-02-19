import argparse
from dataclasses import dataclass
import gzip
import os
import tarfile
import zipfile


@dataclass
class _CliArgs:
    in_file: str
    out_file: str
    action: str


def _parse_args():
    parser = argparse.ArgumentParser(description="Process file to output file")
    parser.add_argument(
        "--in-file",
        type=str,
        required=True,
        help="Input file",
    )

    parser.add_argument("--out-file", type=str, required=True, help="Output file")
    parser.add_argument(
        "--action",
        type=str,
        required=True,
        help="What to do with it",
    )
    args = parser.parse_args()
    return _CliArgs(in_file=args.in_file, out_file=args.out_file, action=args.action)


def is_gzip_file(file_path: str):
    try:
        with gzip.open(file_path, "rb") as f:
            f.read(1)
            return True
    except (gzip.BadGzipFile, OSError):
        return False


def unarchive(file_path: str, target_path: str):
    is_tar_gz = False
    if os.path.exists(target_path):
        print(f"File {target_path} already exists, assuming its extracted already")
        return target_path
    else:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path, "r") as zip_archive:
            zip_archive.extractall(target_path)
    if is_gzip_file(file_path):
        with gzip.open(file_path, "rb") as gzip_archive:
            gzip_extract_path = target_path
            if is_tar_gz:
                gzip_extract_path = f"tarfile_{target_path}"
            with open(gzip_extract_path, "+bw") as gzip_extract_file:
                gzip_extract_file.write(gzip_archive.read())
                if is_tar_gz:
                    with tarfile.open(gzip_extract_path, "r") as tar_file:
                        tar_file.extractall(target_path)
                    os.remove(gzip_extract_path)
    if tarfile.is_tarfile(file_path):
        with tarfile.open(file_path, "r") as tar_file:
            tar_file.extractall(target_path)
    return target_path


if __name__ == "__main__":
    args = _parse_args()
    if args.action == "extract":
        dest = unarchive(args.in_file, args.out_file)
        print(f"Extracted {args.in_file} to {args.out_file}")
