"""
Download helper for VOC 2007/2012 and COCO 2017 train/val.

Usage:
  python scripts/download_data.py --voc --coco --root /Users/shivaninagpal/Documents/Amit/Project/vision/OSOD-impl/data
"""

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlopen
import requests
import shutil
import subprocess

from torchvision.datasets import VOCDetection
from tqdm import tqdm
import tarfile


VOC_YEARS = ["2007", "2012"]
COCO_FILES = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

VOC_TARS = {
    # filename: list of mirror URLs (official first, then mirrors)
    "VOCtrainval_06-Nov-2007.tar": [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
        "https://data.deepai.org/VOCtrainval_06-Nov-2007.tar",
    ],
    "VOCtest_06-Nov-2007.tar": [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar",
        "https://data.deepai.org/VOCtest_06-Nov-2007.tar",
    ],
    "VOCtrainval_11-May-2012.tar": [
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
        "https://data.deepai.org/VOCtrainval_11-May-2012.tar",
    ],
}


def _aria2c_available() -> bool:
    return shutil.which("aria2c") is not None


def _aria2c_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aria2c",
        "-x", "16",
        "-s", "16",
        "-k", "1M",
        "-o", dest.name,
        "-d", str(dest.parent),
        url,
    ]
    print(f"aria2c: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _stream_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Prefer aria2c for speed if available
    if _aria2c_available():
        try:
            _aria2c_download(url, dest)
            return
        except Exception as ex:  # noqa: BLE001
            print(f"aria2c failed ({ex}), falling back to urllib/requests...")

    # Try urllib first
    try:
        with urlopen(url) as resp, open(dest, "wb") as f:
            total = int(resp.headers.get("content-length", 0))
            with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        return
    except Exception:
        # Fall back to requests with a browser user-agent (helps some mirrors with 403)
        headers = {"User-Agent": "Mozilla/5.0"}
        with requests.get(url, stream=True, headers=headers, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def _kaggle_available() -> bool:
    return shutil.which("kaggle") is not None and Path("~/.kaggle/kaggle.json").expanduser().exists()


def _kaggle_download_coco(coco_root: Path) -> None:
    coco_root.mkdir(parents=True, exist_ok=True)
    print("Attempting Kaggle COCO mirror (awsaf49/coco-2017-dataset) ...")
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "awsaf49/coco-2017-dataset",
        "-p",
        str(coco_root),
        "--force",
    ]
    subprocess.run(cmd, check=True)
    archive = coco_root / "coco-2017-dataset.zip"
    if archive.exists():
        _extract_zip(archive, coco_root)


def download_voc(root: Path) -> None:
    voc_root = root / "VOC"

    # First try torchvision helper; if it fails, fallback to mirrors.
    try:
        for year in VOC_YEARS:
            for split in ("trainval", "test") if year == "2007" else ("trainval",):
                print(f"Downloading VOC{year} {split} via torchvision ...")
                VOCDetection(
                    root=voc_root,
                    year=year,
                    image_set=split,
                    download=True,
                )
        print(f"VOC downloaded under {voc_root}")
        return
    except RuntimeError as e:
        print(f"torchvision VOC download failed ({e}). Falling back to mirrors...")

    # Manual mirror download (handle small/blocked downloads; verify size)
    voc_root.mkdir(parents=True, exist_ok=True)
    for fname, urls in VOC_TARS.items():
        tar_path = voc_root / fname
        if tar_path.exists() and tar_path.stat().st_size > 10_000_000:
            print(f"Found existing {fname}, skipping download.")
        else:
            success = False
            for url in urls:
                try:
                    print(f"Attempting {url} ...")
                    _stream_download(url, tar_path)
                    # Basic size sanity check: reject tiny/blocked files (<10 MB)
                    if tar_path.stat().st_size < 10_000_000:
                        print(f"Downloaded file too small ({tar_path.stat().st_size} bytes), trying next mirror...")
                        tar_path.unlink(missing_ok=True)
                        continue
                    success = True
                    break
                except Exception as ex:  # noqa: BLE001
                    print(f"Download failed from {url}: {ex}")
            if not success:
                raise RuntimeError(f"Could not download {fname} from mirrors.")

        extract_dir = voc_root
        print(f"Extracting {fname} ...")
        try:
            with tarfile.open(tar_path, "r") as tf:
                tf.extractall(extract_dir)
        except tarfile.ReadError:
            # Clean up bad file and fail
            tar_path.unlink(missing_ok=True)
            raise RuntimeError(f"Corrupted tarball {tar_path}. Please re-run the script.")

    print(f"VOC downloaded under {voc_root}")


def _extract_zip(path: Path, dest: Path) -> None:
    print(f"Extracting {path.name} ...")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(dest)


def download_coco(root: Path) -> None:
    coco_root = root / "COCO"
    coco_root.mkdir(parents=True, exist_ok=True)

    # If any required files are missing and Kaggle is available, try Kaggle mirror first.
    missing = [fname for fname in COCO_FILES if not (coco_root / fname).exists()]
    if missing and _kaggle_available():
        try:
            _kaggle_download_coco(coco_root)
        except Exception as ex:  # noqa: BLE001
            print(f"Kaggle download failed ({ex}), falling back to official mirrors...")

    for fname, url in COCO_FILES.items():
        zip_path = coco_root / fname
        if not zip_path.exists():
            print(f"Downloading {fname} ...")
            _stream_download(url, zip_path)
        else:
            print(f"Found existing {fname}, skipping download.")

        # Extract if not already extracted
        if fname.startswith("annotations"):
            target_dir = coco_root / "annotations"
        elif fname.startswith("train2017"):
            target_dir = coco_root / "train2017"
        else:
            target_dir = coco_root / "val2017"

        if not target_dir.exists():
            _extract_zip(zip_path, coco_root)
        else:
            print(f"Found extracted folder {target_dir}, skipping extraction.")

    print(f"COCO downloaded under {coco_root}")


def main():
    parser = argparse.ArgumentParser(description="Download VOC and COCO data.")
    parser.add_argument("--root", type=Path, default=Path("data"), help="Root data directory")
    parser.add_argument("--voc", action="store_true", help="Download VOC 2007/2012")
    parser.add_argument("--coco", action="store_true", help="Download COCO 2017 train/val")
    args = parser.parse_args()

    if not args.voc and not args.coco:
        args.voc = True
        args.coco = True

    if args.voc:
        download_voc(args.root)
    if args.coco:
        download_coco(args.root)


if __name__ == "__main__":
    main()
