import json
import shutil
from pathlib import Path
import sys


def remove_image(annotations_path: Path, image_basename: str) -> None:
    # Backup
    backup_path = annotations_path.with_suffix(annotations_path.suffix + ".bak")
    shutil.copy2(annotations_path, backup_path)

    with annotations_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    img_meta = data.get("_via_img_metadata", {})
    img_id_list = data.get("_via_image_id_list", [])

    # Identify image ids and keys to remove
    keys_to_remove = []
    ids_to_remove = []

    # Via stores keys like "R_13.png14687600" with metadata containing filename
    for key, meta in list(img_meta.items()):
        if meta.get("filename") == image_basename:
            keys_to_remove.append(key)

    # Some datasets also store ids in _via_image_id_list matching the same keys
    for img_id in list(img_id_list):
        if isinstance(img_id, str) and img_id.startswith(image_basename):
            ids_to_remove.append(img_id)

    # Remove
    for key in keys_to_remove:
        img_meta.pop(key, None)
    if ids_to_remove:
        data["_via_image_id_list"] = [i for i in img_id_list if i not in ids_to_remove]

    # Write back
    with annotations_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Removed {len(keys_to_remove)} metadata entries and {len(ids_to_remove)} ids for {image_basename}.")
    print(f"Backup saved to: {backup_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/remove_image_from_annotations.py <annotations.json> <image_basename>")
        sys.exit(1)
    annotations_path = Path(sys.argv[1])
    image_basename = sys.argv[2]
    remove_image(annotations_path, image_basename)



