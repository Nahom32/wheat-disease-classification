import os
import zipfile
import argparse

from pathlib import Path


FOLDER_ID = "1Mv2xgnbFT2y8diDKu01nOcP4BpY-7i4F"

# Known CSV file IDs — extracted from the public WFD-2020 dataset.
# These are the train/valid/test CSV splits.
CSV_FILE_IDS = {
    "data_train.csv": None,
    "data_valid.csv": None,
    "data_test.csv": None,
}


def download_folder(folder_id: str, output_dir: str) -> Path:
    """Download an entire Google Drive folder using gdown.

    Returns the path to the downloaded folder.
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download from Google Drive. "
            "Install it with: pip install gdown"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Google Drive folder {folder_id} to {output_path}...")
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url, output=str(output_path), quiet=False)
    print("Folder download complete.")

    return output_path


def download_file(file_id: str, output_path: str) -> str:
    """Download a single file from Google Drive by its file ID."""
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download from Google Drive. "
            "Install it with: pip install gdown"
        )

    url = f"https://drive.google.com/uc?id={file_id}"
    output = gdown.download(url, output=output_path, quiet=False)
    return output


def download_csvs(csv_file_ids: dict, output_dir: str):
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
        download_file(file_id, dest)


def download_wfd_dataset(
    data_root: str = "data/wfd",
    folder_id: str = FOLDER_ID,
    csv_file_ids: dict = None,
    download_images: bool = True,
    download_csv: bool = True,
) -> dict:
    """Download the WFD-2020 dataset from Google Drive.

    Args:
        data_root: Root directory to download into.
        folder_id: Google Drive folder ID containing the images.
        csv_file_ids: Dict mapping CSV filenames to Google Drive file IDs.
        download_images: Whether to download the image folder.
        download_csv: Whether to download the CSV splits.

    Returns:
        Dict with keys 'image_dir' and 'csv_dir' pointing to downloaded paths.
    """
    result = {}

    if download_images:
        images_dir = Path(data_root) / "wfd_dataset"
        if images_dir.exists() and any(images_dir.iterdir()):
            print(f"Image directory {images_dir} already exists and is non-empty, skipping.")
        else:
            download_folder(folder_id, str(images_dir.parent))
        result["image_dir"] = str(images_dir)

    if download_csv:
        csv_dir = Path(data_root) / "csv"
        download_csvs(csv_file_ids or CSV_FILE_IDS, str(csv_dir))
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
    args = parser.parse_args()

    paths = download_wfd_dataset(
        data_root=args.data_root,
        folder_id=args.folder_id,
        download_images=not args.skip_images,
        download_csv=not args.skip_csv,
    )

    print("\nDownload complete. Paths:")
    for key, val in paths.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
