import argparse


def create_args() -> argparse.ArgumentParser:
    """Defines a parameter parser for all of the arguments of the application."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model_str",
        type=str,
        default="decapoda-research/llama-7b-hf",
        help="Pretrained huggingface model to finetune",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=16,
        help="Number of in context examples to use",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-2,
        help="Learning rate",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Number of in context examples to use",
    )

    return parser
