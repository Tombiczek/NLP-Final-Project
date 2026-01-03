import numpy as np
import evaluate
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
)
from tqdm import tqdm
import torch

# Metryki ładowane raz (nie przy każdym wywołaniu funkcji)
ACCURACY = evaluate.load("accuracy")
F1 = evaluate.load("f1")
ROUGE = evaluate.load("rouge")


def baseline_bert_ai_ga(
    ds_ai_ga: DatasetDict,
    model_name: str = "bert-base-uncased",
    output_dir: str = "./bert_baseline_output",
    num_epochs: int = 1,
    batch_size: int = 16,
    max_length: int = 256,
    learning_rate: float = 2e-5,
):
    """
    Trenuje BERT na zbiorze AI-GA i zwraca accuracy oraz F1 (macro).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        return tokenizer(batch["abstract"], truncation=True, max_length=max_length)

    tokenized = ds_ai_ga.map(preprocess, batched=True, remove_columns=["title", "abstract"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": ACCURACY.compute(predictions=preds, references=labels)["accuracy"],
            "f1_macro": F1.compute(predictions=preds, references=labels, average="macro")["f1"],
        }

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",  # Nowa nazwa (evaluation_strategy jest deprecated)
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=50,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,  # Nowa nazwa (tokenizer jest deprecated)
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer.evaluate(tokenized["test"])


def baseline_mt5_zero_shot(
    ds: DatasetDict,
    model_name: str = "google/mt5-small",
    num_samples: int = 200,
    max_input_length: int = 512,
    max_output_length: int = 64,
    batch_size: int = 8,
):
    """
    Zero-shot summarization z mT5. Zwraca metryki ROUGE.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    ds_sample = ds["test"].select(range(min(num_samples, len(ds["test"]))))

    predictions = []
    references = []

    # Batch processing z progress barem
    for i in tqdm(range(0, len(ds_sample), batch_size), desc="Generating summaries"):
        batch = ds_sample[i : i + batch_size]
        
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_input_length,
        ).to(device)

        outputs = model.generate(**inputs, max_length=max_output_length, num_beams=4)
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
        references.extend(batch["summary"])

    return ROUGE.compute(predictions=predictions, references=references)