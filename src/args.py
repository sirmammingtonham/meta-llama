import argparse


def create_args() -> argparse.ArgumentParser:
    """Defines a parameter parser for all of the arguments of the application."""

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Whether or not to run training",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Whether or not to run evaluation",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="data/augment_train_v2",
        help="Directory with the training data",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/baseline_test",
        help="Directory with the training data",
    )
    parser.add_argument(
        "--model_str",
        type=str,
        default="huggyllama/llama-7b",
        help="Pretrained huggingface model to finetune",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=16,
        help="Number of in context examples to use",
    )
    parser.add_argument(
        "--data_method",
        type=str,
        default="direct",
        help="Direct or channel (evaluation with channel still needs to be coded)",
    )
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Pass this flag to use non-augmented data for training",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Where to store the final model.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Number of in context examples to use",
    )

    return parser.parse_args()
