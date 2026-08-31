from __future__ import annotations

import base64
from pathlib import Path


DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tif", ".tiff"}
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
OUTPUT_FILE = BASE_DIR / "image_base64_encodings.txt"


def encode_file_to_base64(file_path: Path) -> str:
	return base64.b64encode(file_path.read_bytes()).decode("ascii")


def collect_image_files(images_dir: Path) -> list[Path]:
	return sorted(
		file_path
		for file_path in images_dir.iterdir()
		if file_path.is_file() and file_path.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS
	)


def write_encodings(image_files: list[Path], output_file: Path) -> None:
	with output_file.open("w", encoding="utf-8") as handle:
		for image_file in image_files:
			handle.write(f"Filename: {image_file.name}\n")
			handle.write(f"Base64: {encode_file_to_base64(image_file)}\n\n")


def main() -> None:
	images_dir = IMAGES_DIR
	output_file = OUTPUT_FILE

	if not images_dir.exists() or not images_dir.is_dir():
		raise FileNotFoundError(f"Image directory not found: {images_dir}")

	image_files = collect_image_files(images_dir)
	if not image_files:
		raise FileNotFoundError(f"No supported image files found in: {images_dir}")

	write_encodings(image_files, output_file)
	print(f"Wrote {len(image_files)} image encodings to {output_file}")


if __name__ == "__main__":
	main()
