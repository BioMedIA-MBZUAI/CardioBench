from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict

SPLIT_REGISTRY: Dict[str, str] = {
    "camus": "cardiobench.workflow.splits.camus:main",
    "echonet_pediatric": "cardiobench.workflow.splits.echonet_pediatric:main",
    "echonet_lvh": "cardiobench.workflow.splits.echonet_lvh_split_tasks:main",
    "hmc_qu": "cardiobench.workflow.splits.hmcqu:main",
    "tmed2_per_image": "cardiobench.workflow.splits.tmed2_1:main",
    "tmed2_per_study": "cardiobench.workflow.splits.tmed2_2:main",
}


def load_callable(target: str) -> Callable[[Path | None], Path]:
    module_name, func_name = target.split(":", 1)
    module = import_module(module_name)
    return getattr(module, func_name)


def run_split(datasets: list[str], config_path: Path | None) -> None:
    for dataset in datasets:
        target = SPLIT_REGISTRY[dataset]
        fn = load_callable(target)
        print(f"-> Generating split for {dataset}...")
        written = fn(config_path)
        print(f"   Output: {written}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Utilities to download datasets and generate CardioBench splits.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser(
        "split", help="Generate dataset splits using the configured paths."
    )
    split_parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(SPLIT_REGISTRY.keys()),
        help="Datasets to split (defaults to all).",
    )
    split_parser.add_argument(
        "--config", type=Path, help="Optional path to a datasets.json configuration."
    )
    split_parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets without running them.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "split":
        datasets = args.datasets or sorted(SPLIT_REGISTRY.keys())
        if args.list:
            print("Available datasets:")
            for name in sorted(SPLIT_REGISTRY.keys()):
                print(f" - {name}")
            return

        run_split(datasets, args.config)


if __name__ == "__main__":
    main()
