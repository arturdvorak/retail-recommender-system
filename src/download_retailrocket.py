import kagglehub
import shutil
from pathlib import Path

# Downloads the RetailRocket ecommerce dataset from Kaggle
# Dataset: retailrocket/ecommerce-dataset
# Copies files to: data/raw/retailrocket

OUTDIR = Path("data/raw/retailrocket")


def ensure_dir(path: Path) -> None:
	"""Create directory if it doesn't exist."""
	path.mkdir(parents=True, exist_ok=True)


def main() -> None:
	# Download latest version of the dataset from Kaggle
	print("Downloading dataset from Kaggle...")
	path = kagglehub.dataset_download("retailrocket/ecommerce-dataset")
	
	print(f"Dataset downloaded to: {path}")
	
	# Create output directory
	ensure_dir(OUTDIR)
	
	# Copy CSV files from kagglehub cache to project directory
	dataset_path = Path(path)
	if dataset_path.exists():
		csv_files = sorted([p for p in dataset_path.glob("*.csv") if p.is_file()])
		
		if not csv_files:
			print("Warning: No CSV files found in downloaded dataset")
			return
		
		print(f"\nCopying {len(csv_files)} files to {OUTDIR.resolve()}...")
		for csv_file in csv_files:
			dest = OUTDIR / csv_file.name
			shutil.copy2(csv_file, dest)
			print(f"  - Copied: {csv_file.name}")
		
		print(f"\nFiles are now available in: {OUTDIR.resolve()}")


if __name__ == "__main__":
	main()
