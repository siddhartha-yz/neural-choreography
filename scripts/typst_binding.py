#!/usr/bin/env python3
"""Minimal Zanim Typst CLI adapter backed by the pinned official Python binding."""

import sys
import typst


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "compile":
        typst.compile(sys.argv[2], output=sys.argv[3], format="svg")
    elif sys.argv[1:] == ["--version"]:
        print("typst 0.15.0 (Python binding)")
    else:
        raise SystemExit("Expected: compile input.typ output.svg")


if __name__ == "__main__":
    main()
