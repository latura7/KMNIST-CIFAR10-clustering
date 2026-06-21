"""Entry point for the KMNIST/CIFAR-10 clustering course project.

Course: Sungkyunkwan University, 2026 Spring,
Machine Learning Cornerstone_ROB5020_41(유현우).
Course page: https://canvas.skku.edu/courses/73956

This project was built based on the course lectures and course-provided
example code, then extended with configurable experiment runs, plots, and
report artifacts.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kmnist_cifar_project.cli import main


if __name__ == "__main__":
    main()
