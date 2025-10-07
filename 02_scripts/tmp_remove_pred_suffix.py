import argparse
import sys
from pathlib import Path


def rename_files(target_dir: Path, recurse: bool = False, dry_run: bool = False) -> int:
    """
    Rename files in the given directory that end with "_pred" before the extension.

    Example: "R_10_pred.h5" -> "R_10.h5"

    - Non-recursive by default; enable recursion with recurse=True
    - If destination exists, the file is skipped to avoid overwrite
    - Returns the number of files renamed
    """
    if not target_dir.exists():
        print(f"[ERROR] Directory does not exist: {target_dir}")
        return 0
    if not target_dir.is_dir():
        print(f"[ERROR] Not a directory: {target_dir}")
        return 0

    iterator = target_dir.rglob("*") if recurse else target_dir.glob("*")

    renamed_count = 0
    for path in iterator:
        if not path.is_file():
            continue

        stem = path.stem
        if not stem.endswith("_pred"):
            continue

        new_stem = stem[:-5]  # remove the trailing "_pred"
        # Preserve all suffixes (e.g., .tar.gz)
        new_name = new_stem + "".join(path.suffixes)
        new_path = path.with_name(new_name)

        if new_path.exists():
            print(f"[SKIP] Destination exists, not overwriting: {new_path}")
            continue

        print(f"[RENAME] {path.name} -> {new_path.name}")
        if not dry_run:
            try:
                path.rename(new_path)
                renamed_count += 1
            except OSError as exc:
                print(f"[ERROR] Failed to rename {path} -> {new_path}: {exc}")

    if dry_run:
        print("[INFO] Dry-run enabled; no files were actually renamed.")

    print(f"[DONE] Files renamed: {renamed_count}")
    return renamed_count


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rename files in a folder by removing the trailing '_pred' before the extension."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help=(
            "Target folder containing files to rename (e.g., BGR/\n"
            "results/5/h5_predictions)."
        ),
    )
    parser.add_argument(
        "--recurse",
        action="store_true",
        help="Recurse into subdirectories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be renamed without making changes.",
    )

    args = parser.parse_args(argv)
    rename_files(args.directory, recurse=args.recurse, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


