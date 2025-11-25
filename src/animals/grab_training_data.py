import os
import re
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# FINAL TARGET DIRECTORIES
ROOT = Path("src/animals/images")
ROOT.mkdir(parents=True, exist_ok=True)

def parse_kaggle_link(url: str) -> str:
    """
    Accept Kaggle links like:
      https://www.kaggle.com/datasets/tongpython/cat-and-dog
    And return 'tongpython/cat-and-dog'
    """
    match = re.search(r"kaggle\.com/datasets/([^/]+)/([^/?]+)", url)
    if not match:
        raise ValueError("Invalid Kaggle dataset link: " + url)
    owner, dataset = match.groups()
    return f"{owner}/{dataset}"

def kaggle_download(dataset_link: str, dataset_name: str):
    """
    Download a Kaggle dataset (given a link OR owner/dataset) into:
    src/animals/images/<dataset_name>/
    """
    target_dir = ROOT / dataset_name

    if target_dir.exists():
        print(f"[skip] {dataset_name} already downloaded.")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)

    # Convert Kaggle link → dataset identifier
    if "kaggle.com" in dataset_link:
        dataset_id = parse_kaggle_link(dataset_link)
    else:
        dataset_id = dataset_link  # already in owner/dataset form

    print(f"[download] Kaggle dataset: {dataset_id}")

    cmd = [
        "kaggle", "datasets", "download",
        "-d", dataset_id,
        "-p", str(target_dir),
        "--unzip"
    ]
    subprocess.run(cmd, check=True)

    print(f"[done] Download complete: {dataset_name}")
    return target_dir

def normalize_structure(dataset_dir: Path):
    """
    Ensure folder structure:

    src/animals/images/<dataset_name>/
        training_set/
            cats/
            dogs/
        test_set/
            cats/
            dogs/
    """

    # Find actual training/test directories inside downloaded content
    for sub in dataset_dir.glob("**/*"):
        if sub.is_dir() and "train" in sub.name.lower():
            train_src = sub
        if sub.is_dir() and "test" in sub.name.lower():
            test_src = sub

    train_dest = dataset_dir / "training_set"
    test_dest = dataset_dir / "test_set"

    # Clear existing auto-created dirs if needed
    if train_dest.exists(): shutil.rmtree(train_dest)
    if test_dest.exists(): shutil.rmtree(test_dest)

    train_dest.mkdir()
    test_dest.mkdir()

    print("[reformat] Cleaning and moving into standard structure")

    # Move data into correct place
    shutil.move(str(train_src), str(train_dest / "training_set"))
    shutil.move(str(test_src), str(test_dest / "test_set"))

    print("[done] Normalized dataset structure.")

def import_kaggle_dataset(url: str, dataset_name: str):
    dataset_dir = kaggle_download(url, dataset_name)
    normalize_structure(dataset_dir)

# -------------------------------
# Example Usage
# -------------------------------

def main():
    import_kaggle_dataset(
        "https://www.kaggle.com/datasets/tongpython/cat-and-dog",
        "cats_dogs"
    )

if __name__ == "__main__":
    main()

