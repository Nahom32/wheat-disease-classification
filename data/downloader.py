import os
import time
import zipfile
import argparse

from pathlib import Path


FOLDER_ID = "1Mv2xgnbFT2y8diDKu01nOcP4BpY-7i4F"

# Known CSV file IDs — extracted from the public WFD-2020 dataset.
# These are the train/valid/test CSV splits.
CSV_FILE_IDS = {
    "data_train.csv": "1aRt-vOy-au59yGS7rdbRTI2wKWa7QKfZ",
    "data_valid.csv": "1EbAa1m32FWK2I6asBw5gZzwaiCebJRqm",
    "data_test.csv": "1L8AIshTV1ZnXYG9EIo-Db62VQptFFWQg",
}

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_RETRY_DELAY = 5.0
MAX_RETRY_DELAY = 60.0


def _import_gdown():
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download from Google Drive. "
            "Install it with: pip install gdown"
        )
    return gdown


def _run_with_retries(func, description, max_attempts, retry_delay):
    """Run func() with retries and exponential backoff.

    Google Drive throttles large downloads, so a single attempt may be
    interrupted. This loop re-runs func until it succeeds or max_attempts
    is exhausted, sleeping longer between each retry.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            result = func()
            if result:
                return result
            print(f"{description} returned no output (likely throttled).")
        except Exception as exc:
            print(f"{description} failed on attempt {attempt}/{max_attempts}: {exc}")

        if attempt == max_attempts:
            break
        delay = min(retry_delay * (2 ** (attempt - 1)), MAX_RETRY_DELAY)
        print(f"Retrying {description} in {delay:.1f}s...")
        time.sleep(delay)

    raise RuntimeError(f"{description} failed after {max_attempts} attempts.")


def download_file(
    file_id: str,
    output_path: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> str:
    """Download a single file from Google Drive by its file ID.

    Retries with exponential backoff to recover from Google Drive throttling
    large downloads. Passes resume=True to gdown so retries continue from a
    partially downloaded temp file instead of restarting.
    """
    gdown = _import_gdown()
    url = f"https://drive.google.com/uc?id={file_id}"

    def _attempt():
        # gdown returns the output path on success and None on failure
        # (e.g. throttled/truncated downloads).
        result = gdown.download(url, output=output_path, quiet=False, resume=True)
        if result is None or not os.path.exists(result) or os.path.getsize(result) == 0:
            return None
        return result

    return _run_with_retries(_attempt, f"Downloading file {file_id}", max_attempts, retry_delay)


def _list_folder_files(folder_id: str):
    """Enumerate files in a Google Drive folder.

    Returns a list of (file_id, relative_path) tuples. Uses gdown's internal
    folder-parsing helpers so files can be downloaded individually (and retried
    independently); raises if the internal API is unavailable.
    """
    from gdown.download_folder import (
        _download_and_parse_google_drive_link,
        _get_directory_structure,
    )
    from gdown.download import _get_session

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    sess, _ = _get_session(
        proxy=None,
        use_cookies=True,
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/98.0.4758.102 Safari/537.36"
        ),
    )
    return_code, gdrive_file = _download_and_parse_google_drive_link(
        sess, url, quiet=True, verify=True
    )
    if not return_code:
        raise RuntimeError(f"Could not retrieve folder contents for {folder_id}.")
    structure = _get_directory_structure(gdrive_file, "")
    return [(file_id, path) for file_id, path in structure if file_id is not None]


def _download_folder_via_gdown(
    gdown, folder_id: str, output_path: Path, max_attempts: int, retry_delay: float
) -> Path:
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    def _attempt():
        return gdown.download_folder(url, output=str(output_path), quiet=False)

    _run_with_retries(_attempt, f"Downloading folder {folder_id}", max_attempts, retry_delay)
    return output_path


def download_folder(
    folder_id: str,
    output_dir: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> Path:
    """Download an entire Google Drive folder using gdown.

    Each file is downloaded individually so that a throttled file is retried on
    its own rather than restarting the whole folder. Falls back to retrying
    gdown.download_folder if the folder cannot be enumerated (e.g. >50 files).

    Returns the path to the downloaded folder.
    """
    gdown = _import_gdown()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Google Drive folder {folder_id} to {output_path}...")
    try:
        files = _list_folder_files(folder_id)
    except Exception as exc:
        print(
            f"Could not enumerate folder contents ({exc}); "
            "falling back to gdown.download_folder with retries."
        )
        return _download_folder_via_gdown(gdown, folder_id, output_path, max_attempts, retry_delay)

    for file_id, rel_path in files:
        dest = output_path / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            print(f"{dest} already exists, skipping.")
            continue
        download_file(file_id, str(dest), max_attempts=max_attempts, retry_delay=retry_delay)
    print("Folder download complete.")

    return output_path


def download_csvs(csv_file_ids: dict, output_dir: str, **retry_kwargs):
    """Download CSV files. Skips entries where the ID is None."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, file_id in csv_file_ids.items():
        if file_id is None:
            print(f"Skipping {name} — no file ID provided.")
            continue
        dest = str(output_path / name)
        if os.path.exists(dest):
            print(f"{name} already exists at {dest}, skipping.")
            continue
        download_file(file_id, dest, **retry_kwargs)


def download_wfd_dataset(
    data_root: str = "data/wfd",
    folder_id: str = FOLDER_ID,
    csv_file_ids: dict = None,
    download_images: bool = True,
    download_csv: bool = True,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> dict:
    """Download the WFD-2020 dataset from Google Drive.

    Args:
        data_root: Root directory to download into.
        folder_id: Google Drive folder ID containing the images.
        csv_file_ids: Dict mapping CSV filenames to Google Drive file IDs.
        download_images: Whether to download the image folder.
        download_csv: Whether to download the CSV splits.
        max_attempts: Max times to retry a throttled download.
        retry_delay: Initial delay (seconds) before the first retry; doubles
            with each subsequent attempt.

    Returns:
        Dict with keys 'image_dir' and 'csv_dir' pointing to downloaded paths.
    """
    result = {}

    if download_images:
        images_dir = Path(data_root) / "wfd_dataset"
        if images_dir.exists() and any(images_dir.iterdir()):
            print(f"Image directory {images_dir} already exists and is non-empty, skipping.")
        else:
            download_folder(
                folder_id,
                str(images_dir.parent),
                max_attempts=max_attempts,
                retry_delay=retry_delay,
            )
        result["image_dir"] = str(images_dir)

    if download_csv:
        csv_dir = Path(data_root) / "csv"
        download_csvs(
            csv_file_ids or CSV_FILE_IDS,
            str(csv_dir),
            max_attempts=max_attempts,
            retry_delay=retry_delay,
        )
        result["csv_dir"] = str(csv_dir)

    return result


def main():
    parser = argparse.ArgumentParser(description="Download WFD-2020 dataset from Google Drive")
    parser.add_argument("--data-root", default="data/wfd", help="Destination directory")
    parser.add_argument("--folder-id", default=FOLDER_ID,
                        help="Google Drive folder ID containing images")
    parser.add_argument("--skip-images", action="store_true",
                        help="Skip downloading the image folder")
    parser.add_argument("--skip-csv", action="store_true",
                        help="Skip downloading the CSV files")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help="Max times to retry a throttled download")
    parser.add_argument("--retry-delay", type=float, default=DEFAULT_RETRY_DELAY,
                        help="Initial retry delay in seconds (doubles each retry)")
    args = parser.parse_args()

    paths = download_wfd_dataset(
        data_root=args.data_root,
        folder_id=args.folder_id,
        download_images=not args.skip_images,
        download_csv=not args.skip_csv,
        max_attempts=args.max_attempts,
        retry_delay=args.retry_delay,
    )

    print("\nDownload complete. Paths:")
    for key, val in paths.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
