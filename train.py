import math
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from src.data import load_datasets, ICLCollator
from src.args import create_args
from src.model import ICLModel
from src.config import TRAIN_TASKS, TEST_TASKS
from src.trainer import ICLTrainer


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

    training_args = TrainingArguments(
        output_dir="./models",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        logging_dir="./logs",
        report_to="wandb",
        remove_unused_columns=False,
        save_total_limit=5,
        seed=args.seed,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    trainer = ICLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=val_datasets,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics = trainer.evaluate()

    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    trainer.create_model_card()

    return trainer


if __name__ == "__main__":
    args = create_args()
    train(args)
