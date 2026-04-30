import os
import requests
from urllib.parse import urlparse
import argparse
from dataclasses import dataclass


@dataclass
class _CliArgs:
    url: str
    filename: str | None = None
    base_dir: str = "data/downloads"


def _parse_args():
    parser = argparse.ArgumentParser(description="Download file from URL")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/downloads",
        required=False,
        help="Base directory where the file will be saved",
    )

    parser.add_argument(
        "--url", type=str, required=True, help="URL of the file to download"
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default=None,
        required=False,
        help="Filename to give to downloaded file (will try to derive from url if not given)",
    )

    args = parser.parse_args()
    return _CliArgs(url=args.url, base_dir=args.base_dir, filename=args.file_name)


def download_file_from_url(url: str, file_path: str):
    if os.path.exists(file_path):
        print(
            f"File {file_path} already exists and {url} does not provide content-length header. Assuming this is already downloaded and skipping"
        )
        return file_path
    with open(file_path, "wb") as target_file:
        res = requests.get(url, stream=True)
        res.raise_for_status()
        target_file.write(res.content)
    print(f"Downloaded {file_path} from {url}")
    return file_path


def download_file_in_chunks(url: str, content_length: int, file_path: str):
    # Check if file already exists
    if os.path.exists(file_path):
        existing_size = os.path.getsize(file_path)
        if existing_size == content_length:
            print(
                f"File '{file_path}' already exists with correct size ({content_length} bytes). Skipping download."
            )
            return file_path
        else:
            print(
                f"File exists but size mismatch ({existing_size} vs {content_length}). Deleting and restarting."
            )
            os.remove(file_path)

    # Download file in chunks
    print(f"Downloading '{file_path}' ({content_length} bytes)...")
    downloaded = 0
    chunk_size = 8192

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                progress = (downloaded / content_length) * 100
                print(
                    f"Progress: {downloaded}/{content_length} bytes ({progress:.1f}%)",
                    end="\r",
                )

    print(f"\nDownload complete: '{file_path}'")
    return file_path


def download_file(url: str, base_dir: str, filename: str | None = None):
    # Extract filename from URL
    parsed_url = urlparse(url)
    if not filename:
        filename = os.path.basename(parsed_url.path)
        if not filename:
            raise RuntimeError(
                "Could not determine filename from url and filename argument was None"
            )

    file_path = f"{base_dir}/{filename}"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # Get content length via HEAD request
    head_response = requests.head(
        url, allow_redirects=True, headers={"Accept-Encoding": "*/*"}
    )
    head_response.raise_for_status()

    content_length = int(head_response.headers.get("Content-Length", 0))
    if content_length == 0:
        print(
            f"Url {url} did not provide Content-Length header, download progress will not be possible to monitor"
        )
        return download_file_from_url(url, file_path)
    else:
        return download_file_in_chunks(url, content_length, file_path)


if __name__ == "__main__":
    args = _parse_args()
    file_path = download_file(args.url, args.base_dir, args.filename)
    print(f"Downloaded: {file_path}")
