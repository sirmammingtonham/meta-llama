import accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from src.data import load_datasets, ICLCollator, MultitaskDataloader
from src.args import create_args
from src.model import ICLModel
from src.config import TRAIN_TASKS, TEST_TASKS

from torch.utils.data import DataLoader


def train(args):
    # setup models and peft
    base_model = AutoModelForCausalLM.from_pretrained(args.model_str)
    tokenizer = AutoTokenizer.from_pretrained(args.model_str)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    base_model = get_peft_model(base_model, peft_config)
    base_model.print_trainable_parameters()

    model = ICLModel(base_model, k_examples=args.k)


    # setup datasets and dataloaders
    train_datasets = load_datasets(TRAIN_TASKS)
    val_datasets = load_datasets(TEST_TASKS)

    collate_fn = ICLCollator(tokenizer, k_examples=args.k)
    loader_batch_size = args.k * args.batch_size

    train_dataloader_dict = {
        k: DataLoader(v, batch_size=loader_batch_size, collate_fn=collate_fn)
        for k, v in train_datasets.items()
    }
    val_dataloader_dict = {
        k: DataLoader(v, batch_size=loader_batch_size, collate_fn=collate_fn)
        for k, v in val_datasets.items()
    }

    train_dataloader = MultitaskDataloader(train_dataloader_dict)
    val_dataloader = MultitaskDataloader(val_dataloader_dict)

    # TODO: custom training loop using accelerate

    return


if __name__ == "__main__":
    args = create_args()
    train(args)
