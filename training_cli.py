"""Shared command-line helpers for training entry points.

This module keeps the three experiment scripts aligned while preserving
their script-specific defaults.
"""

from __future__ import annotations

import argparse
import ast
from typing import Any, Callable


DEFAULT_NUM_ITERATION = ((1, 0), (1, 0), (1, 0), (1, 1), (2, 0))
SAMPLE_RATE_OVERRIDES = {
    "darcy": 2,
    "darcy20c6": 2,
    "darcy15c10": 2,
    "darcyF": 2,
    "darcy_contin": 2,
    "a4f1": 4,
    "helm": 1,
    "pipe": 1,
}


def build_training_parser(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    loss_type: str,
    model_type_help: str,
) -> argparse.ArgumentParser:
    """Create the shared CLI parser used by all training scripts."""

    parser = argparse.ArgumentParser(
        description="Train a Newton-Informed Neural Operator experiment."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="ns_merge",
        help="data name, darcy, darcy20c6, darcy15c10, darcyF, darcy_contin",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="DeepONet",
        help=model_type_help,
    )
    parser.add_argument("--epochs", type=int, default=epochs, help="number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=batch_size, help="batch size"
    )
    parser.add_argument(
        "--optimizer_type", type=str, default="adam", help="optimizer type"
    )
    parser.add_argument("--lr", type=float, default=lr, help="learning rate")
    parser.add_argument(
        "--final_div_factor", type=float, default=10, help="final_div_factor"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay"
    )
    parser.add_argument(
        "--loss_type", type=str, default=loss_type, help="loss type, l2, h1, pde"
    )
    parser.add_argument("--GN", action="store_true", help="use normalized x")
    parser.add_argument("--sample_x", action="store_true", help="sample x")
    parser.add_argument(
        "--sampling_rate", type=int, default=1, help="sampling rate"
    )
    parser.add_argument("--normalizer", action="store_true", help="use normalizer")
    parser.add_argument(
        "--normalizer_type", type=str, default="GN", help="PGN, GN"
    )
    parser.add_argument("--num_layer", type=int, default=5, help="number of layers")
    parser.add_argument(
        "--num_channel_u", type=int, default=24, help="number of channels for u"
    )
    parser.add_argument(
        "--num_channel_f", type=int, default=1, help="number of channels for f"
    )
    parser.add_argument(
        "--num_iteration",
        type=str,
        nargs="+",
        default=[str(list(values)) for values in DEFAULT_NUM_ITERATION],
        help="number of iterations in each layer, e.g. --num_iteration '[1,0]' '[2,1]'",
    )
    parser.add_argument(
        "--padding_mode", type=str, default="zeros", help="padding mode"
    )
    parser.add_argument(
        "--last_layer", type=str, default="linear", help="last layer type"
    )
    parser.add_argument("--test", action="store_true", help="load model and test")
    parser.add_argument(
        "--MODEL_PATH_LOAD", type=str, default="PATH_LOAD", help="PATH_LOAD"
    )
    return parser


def _normalize_num_iteration(raw_iterations: list[Any]) -> list[list[int]]:
    """Convert CLI values into a nested list of integers."""

    normalized = []
    for iteration in raw_iterations:
        parsed_iteration: Any = iteration
        if isinstance(iteration, str):
            parsed_iteration = ast.literal_eval(iteration)
        normalized.append([int(value) for value in parsed_iteration])
    return normalized


def parse_training_args(
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    loss_type: str,
    model_type_help: str,
    argv: list[str] | None = None,
) -> dict[str, Any]:
    """Parse, normalize, and post-process shared training arguments."""

    parser = build_training_parser(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        loss_type=loss_type,
        model_type_help=model_type_help,
    )
    args = vars(parser.parse_args(argv))
    args["num_iteration"] = _normalize_num_iteration(args["num_iteration"])

    if args["sample_x"]:
        args["sampling_rate"] = SAMPLE_RATE_OVERRIDES.get(
            args["data"], args["sampling_rate"]
        )

    return args


def build_training_configs(
    args: dict[str, Any],
    data_size_getter: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create data, model, and optimizer dictionaries from parsed args."""

    data_options = {
        "data": args["data"],
        "sampling_rate": args["sampling_rate"],
        "sample_x": args["sample_x"],
        "batch_size": args["batch_size"],
        "loss_type": args["loss_type"],
        "loss_weight": [2],
        "normalizer_type": args["normalizer_type"],
        "GN": args["GN"],
        "MODEL_PATH_LOAD": args["MODEL_PATH_LOAD"],
    }
    data_options = data_size_getter(data_options)

    model_options = {
        "num_layer": args["num_layer"],
        "num_channel_u": args["num_channel_u"],
        "num_channel_f": args["num_channel_f"],
        "num_classes": 1,
        "num_iteration": args["num_iteration"],
        "in_chans": 1,
        "normalizer": args["normalizer"],
        "output_dim": 1,
        "activation": "gelu",
        "padding_mode": args["padding_mode"],
        "last_layer": args["last_layer"],
    }

    optimizer_options = {
        "optimizer_type": args["optimizer_type"],
        "lr": args["lr"],
        "weight_decay": args["weight_decay"],
        "epochs": args["epochs"],
        "final_div_factor": args["final_div_factor"],
        "div_factor": 2,
    }

    return data_options, model_options, optimizer_options
