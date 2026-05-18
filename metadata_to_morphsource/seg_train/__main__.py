"""Entry-point so ``python -m metadata_to_morphsource.seg_train ...`` works."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
