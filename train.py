import argparse
import json
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model
from feedback_summarizer.prompts import make_prompt

BASE_MODEL = "google/flan-t5-small"

def load_examples(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if not item.get("feedback") or not item.get("summary"):
                    raise ValueError("Each row needs feedback and summary.")
                rows.append(item)
    if len(rows) < 2:
        raise ValueError("Add at least two feedback-summary examples.")
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/feedback.jsonl")
    parser.add_argument("--output", default="outputs/lora_adapter")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    rows = load_examples(args.data)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=16,
        lora_dropout=0.1, target_modules=["q", "v"]
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    def tokenize(batch):
        inputs = tokenizer([make_prompt(x) for x in batch["feedback"]],
                           max_length=512, truncation=True)
        labels = tokenizer(text_target=batch["summary"], max_length=128, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    dataset = Dataset.from_list(rows).map(tokenize, batched=True, remove_columns=["feedback", "summary"])
    training_args = TrainingArguments(
        output_dir="outputs/training_checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="epoch",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, model=model))
    trainer.train()
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"LoRA adapter saved to {args.output}")

if __name__ == "__main__":
    main()
