import pyarrow.parquet as pq
import csv
from pathlib import Path

# Dataset location
BASE = Path("data/raw/hagrid_extracted/hagrid-sample-30k-384p")
ANNOTATION_FILE = BASE / "annotations_df.parquet"

# TouchFree classes
TARGET_CLASSES = {
    "palm",
    "fist",
    "like",
    "dislike",
    "one",
    "peace",
    "no_gesture",
}

# Read annotations
table = pq.read_table(ANNOTATION_FILE)

labels = table["labels"].to_pylist()
user_ids = table["user_id"].to_pylist()
image_ids = table["__index_level_0__"].to_pylist()

rows = []

for image_id, label_list, user_id in zip(image_ids, labels, user_ids):

    if not label_list:
        continue

    # Use the main/first label
    label = label_list[0]

    if label not in TARGET_CLASSES:
        continue

    rows.append([
        image_id,
        label,
        user_id
    ])

# Create output folder
OUTPUT_DIR = Path("data/manifests")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "touchfree_manifest.csv"

# Write CSV without pandas
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "image_id",
        "label",
        "user_id"
    ])

    writer.writerows(rows)

print("===================================")
print("TouchFree Manifest Created!")
print("===================================")

print(f"Images: {len(rows)}")
print(f"Saved to: {OUTPUT_FILE}")

# Count classes
class_counts = {}

for row in rows:
    label = row[1]
    class_counts[label] = class_counts.get(label, 0) + 1

print("\nClass counts:")

for label in sorted(class_counts):
    print(f"{label}: {class_counts[label]}")

# Count users
unique_users = set(row[2] for row in rows)

print(f"\nUnique users: {len(unique_users)}")