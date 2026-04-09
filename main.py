from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.core.predictor import Predictor
from src.core.trainer import Trainer


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ForecastLab-AI CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a forecasting model")
    train_parser.add_argument(
        "--hyperparameters",
        required=True,
        type=str,
        help="Path to hyperparameters json file",
    )
    train_parser.add_argument(
        "--data",
        required=True,
        type=str,
        help="Path to training data file (.csv or .parquet)",
    )
    train_parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output artifact path (.pkl, .csv, .parquet)",
    )

    predict_parser = subparsers.add_parser("predict", help="Run predictions from a trained artifact")
    predict_parser.add_argument(
        "--artifact",
        required=True,
        type=str,
        help="Path to trained artifact (.pkl)",
    )
    predict_parser.add_argument(
        "--data",
        type=str,
        help="Path to prediction input file (.csv or .parquet)",
    )
    predict_parser.add_argument(
        "--periods",
        type=int,
        help="Future periods for Prophet prediction",
    )
    predict_parser.add_argument(
        "--include-history",
        action="store_true",
        help="Include training history for Prophet predictions",
    )
    predict_parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output predictions path (.csv, .parquet, .pkl)",
    )

    return parser


def run_train(args: argparse.Namespace) -> None:
    trainer = Trainer.from_hyperparameters_file(Path(args.hyperparameters))
    artifact = trainer.train_from_file(Path(args.data))
    trainer.save_output(artifact, Path(args.output))
    LOGGER.info("Training completed.")


def run_predict(args: argparse.Namespace) -> None:
    predictor = Predictor.from_file(Path(args.artifact))
    predictions = predictor.predict_from_file(
        data_path=Path(args.data) if args.data else None,
        periods=args.periods,
        include_history=args.include_history,
    )
    predictor.save_predictions(predictions, Path(args.output))
    LOGGER.info("Prediction completed.")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    else:
        parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()