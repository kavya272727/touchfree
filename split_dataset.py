import csv
import random
from pathlib import Path
from collections import defaultdict, Counter

MANIFEST = Path("data/manifests/touchfree_manifest.csv")
OUTPUT_DIR = Path("data/manifests")

random.seed(42)

# -----------------------------
# Load manifest
# -----------------------------
rows = []

with open(MANIFEST, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print("Total images:", len(rows))

# -----------------------------
# Group images by user
# -----------------------------
users = defaultdict(list)

for row in rows:
    users[row["user_id"]].append(row)

user_ids = list(users.keys())
random.shuffle(user_ids)

print("Unique users:", len(user_ids))

# -----------------------------
# Split users
# -----------------------------
n = len(user_ids)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_users = set(user_ids[:train_end])
val_users = set(user_ids[train_end:val_end])
test_users = set(user_ids[val_end:])

# -----------------------------
# Assign images
# -----------------------------
train_rows = []
val_rows = []
test_rows = []

for row in rows:
    user = row["user_id"]

    if user in train_users:
        row["split"] = "train"
        train_rows.append(row)

    elif user in val_users:
        row["split"] = "val"
        val_rows.append(row)

    else:
        row["split"] = "test"
        test_rows.append(row)

# -----------------------------
# Save split manifest
# -----------------------------
output_file = OUTPUT_DIR / "touchfree_split_manifest.csv"

fieldnames = list(rows[0].keys())

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(train_rows)
    writer.writerows(val_rows)
    writer.writerows(test_rows)

# -----------------------------
# Print results
# -----------------------------
print("\n===================================")
print("TouchFree Dataset Split Complete!")
print("===================================")

print("Train images:", len(train_rows))
print("Validation images:", len(val_rows))
print("Test images:", len(test_rows))

print("\nUsers:")
print("Train users:", len(train_users))
print("Validation users:", len(val_users))
print("Test users:", len(test_users))

print("\nClass distribution:")

for split_name, split_rows in [
    ("TRAIN", train_rows),
    ("VAL", val_rows),
    ("TEST", test_rows)
]:
    counts = Counter(row["label"] for row in split_rows)

    print(f"\n{split_name}:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

print("\nSaved to:")
print(output_file)