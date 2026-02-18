import argparse
from pathlib import Path


def _is_legacy_classifier_artifact(path: Path) -> bool:
    name = path.name
    if name.startswith("random_forest_"):
        return not name.startswith("random_forest_global")
    if name.startswith("xgboost_"):
        return not name.startswith("xgboost_global")
    return False


def find_legacy_artifacts(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []
    candidates = [p for p in models_dir.iterdir() if p.is_file() and _is_legacy_classifier_artifact(p)]
    return sorted(candidates, key=lambda p: p.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove legacy per-ticker RF/XGBoost artifacts. "
            "Global artifacts (random_forest_global*, xgboost_global*) are preserved."
        )
    )
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply deletion. Without this flag, script performs a dry-run only.",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    targets = find_legacy_artifacts(models_dir)

    print(f"Models directory: {models_dir}")
    print(f"Legacy RF/XGB artifacts found: {len(targets)}")
    if not targets:
        print("Nothing to clean.")
        return

    for path in targets:
        print(path.as_posix())

    if not args.apply:
        print("\nDry-run complete. Re-run with --apply to delete the files above.")
        return

    deleted = 0
    for path in targets:
        path.unlink(missing_ok=True)
        deleted += 1

    print(f"\nDeleted {deleted} files.")


if __name__ == "__main__":
    main()
