import os
import requests
from urllib.parse import urlparse
import argparse
from dataclasses import dataclass
import zipfile
import gzip
import tarfile


@dataclass
class CliArgs:
    url: str
    chunk_size = 1024 * 1024
    filename: str | None = None
    base_dir: str = "data/downloads"


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


def download_file_in_chunks(
    url: str, content_length: int, file_path: str, chunk_size: int
):
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

    with open(file_path, "wb") as f:
        while downloaded < content_length:
            # Calculate range for this chunk
            start = downloaded
            end = min(downloaded + chunk_size - 1, content_length - 1)

            # Set Content-Range header
            headers = {"Range": f"bytes={start}-{end}"}

            # Download chunk
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()

            # Write chunk to file
            chunk_data = response.content
            f.write(chunk_data)

            downloaded += len(chunk_data)

            # Print progress
            progress = (downloaded / content_length) * 100
            print(
                f"Progress: {downloaded}/{content_length} bytes ({progress:.1f}%)",
                end="\r",
            )

    print(f"\nDownload complete: '{file_path}'")
    return file_path


def download_file(
    url: str, base_dir: str, filename: str | None = None, chunk_size=1024 * 1024
):
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
        return download_file_in_chunks(url, content_length, file_path, chunk_size)


def is_gzip_file(file_path: str):
    try:
        with gzip.open(file_path, "rb") as f:
            f.read(1)
            return True
    except (gzip.BadGzipFile, OSError):
        return False


def unarchive(file_path: str, target_path: str | None = None):
    is_tar_gz = False
    if target_path == None:
        base_name = file_path.split(os.path.sep).pop()
        if base_name.endswith(".zip"):
            target_path = base_name.replace(".zip", "")
        if base_name.endswith(".gz"):
            target_path = base_name.replace(".gz", "")
        if base_name.endswith(".tar.gz"):
            is_tar_gz = True
            target_path = base_name.replace(".tar.gz", "")
    if target_path == None:
        raise ValueError(
            f"Could not determine target file path from {file_path} and given argument was None"
        )
    if os.path.exists(target_path):
        print(f"File {target_path} already exists, assuming its extracted already")
        return target_path
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


def parse_args():
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
    return CliArgs(url=args.url, base_dir=args.base_dir, filename=args.file_name)


if __name__ == "__main__":
    args = parse_args()
    file_path = download_file(args.url, args.base_dir, args.filename, args.chunk_size)
    extracted_file = unarchive(file_path)
    print(f"Extracted: {extracted_file}")
