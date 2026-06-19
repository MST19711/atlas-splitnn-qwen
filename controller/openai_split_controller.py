#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from controller.openai_controller import build_app, main

__all__ = ["build_app", "main"]


if __name__ == "__main__":
    main()
