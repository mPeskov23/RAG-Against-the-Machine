"""Entry point for python -m src CLI."""

import fire
from .cli import RagCLI


def main() -> None:
    """Run CLI interface using Python Fire."""
    fire.Fire(RagCLI)


if __name__ == "__main__":
    main()
