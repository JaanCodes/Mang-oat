"""
Script to scale the 'Production' column in `submission2.csv` by 4 and round up to the nearest integer.
This script will:
 - Make a backup of the original `submission2.csv` as `submission2_backup.csv`.
 - Overwrite `submission2.csv` with the scaled/rounded values.
 - Create a separate file `submission2_x4.csv` with the scaled/rounded values as well.

Usage:
  python py.py
Outputs:
  - submission2_backup.csv (original file)
  - submission2.csv (updated in place)
  - submission2_x4.csv (new file with scaled values)
"""

import csv
import math
import random
from pathlib import Path


def scale_submission(input_path: Path, output_path: Path, inplace_backup: Path, random_seed=None):
	# Read input CSV
	rows = []
	header = None
	with input_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.reader(f)
		for i, row in enumerate(reader):
			if i == 0:
				header = row
				continue
			rows.append(row)

	# Ensure header contains expected column name
	if header is None or len(header) < 2:
		raise RuntimeError("Unexpected CSV format for submission file")

	# Find index for production column
	prod_col = None
	for idx, name in enumerate(header):
		if str(name).strip().lower() in ("production", "demand"):
			prod_col = idx
			break
	if prod_col is None:
		# Default to second column if we can't find the name
		prod_col = 1

	# Create backup (ensure we don't overwrite existing backup)
	bak = inplace_backup
	i = 0
	while bak.exists():
		i += 1
		bak = inplace_backup.with_name(inplace_backup.stem + f"_{i}" + inplace_backup.suffix)
	input_path.replace(bak)  # Moves original to backup path

	# Create transformed rows and write to both output files
	transformed_rows = []
	if random_seed is not None:
		random.seed(random_seed)
	for row in rows:
		# Keep ID as is, multiply production by 4 and ceil
		try:
			prod_val = float(row[prod_col])
		except Exception:
			# If conversion fails, just keep as-is
			transformed_rows.append(row)
			continue
		rand_sub = random.randint(0, 100)
		scaled_val = prod_val * 4 - rand_sub
		# Make sure we don't go negative
		scaled_val = max(scaled_val, 0)
		scaled = math.ceil(scaled_val)
		new_row = list(row)
		new_row[prod_col] = str(int(scaled))
		transformed_rows.append(new_row)

	# Write transformed to the main output (overwrite original name)
	with output_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(header)
		writer.writerows(transformed_rows)


def main():
	cwd = Path('.').resolve()
	input_file = cwd / 'submission2.csv'
	backup_file = cwd / 'submission2_backup.csv'
	x4_file = cwd / 'submission2_x4_rand.csv'

	if not input_file.exists():
		print(f"File not found: {input_file}")
		return

	# Make a copy and then scale
	# Use a seed for reproducibility if needed; default None yields random behavior
	seed = None
	scale_submission(input_file, input_file, backup_file, random_seed=seed)
	# Also create the separate x4 file from the backup (the backup still contains original values)
	scale_submission(backup_file, x4_file, backup_file.with_name('submission2_backup2.csv'), random_seed=seed)

	print("Done: created scaled files:")
	print(f" - Backup: {backup_file}")
	print(f" - Overwrote: {input_file}")
	print(f" - New file: {x4_file}")


if __name__ == '__main__':
	main()

