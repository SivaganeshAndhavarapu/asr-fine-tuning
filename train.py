
import os
import torch
from dataclasses import dataclass
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer

# ----------- Load and Prepare Dataset -------------
csv_path = "/home/hp/sivaganesh/ada_test/train.csv"
dataset = load_dataset("csv", data_files={"train": csv_path}, split="train")

# Load audio from 'path' column and resample to 16kHz
dataset = dataset.cast_column("path", Audio(sampling_rate=16000))
dataset = dataset.filter(lambda example: example["text"] and example["text"].strip() != "")
# ----------- Load Whisper Model and Processor -------------
model_name = "openai/whisper-base.en" # for other model need to change
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# ----------- Preprocessing Function -------------
def prepare_example(example):
    audio = example["path"]  # or "audio" depending on your column name
    example["input_features"] = processor(audio["array"], sampling_rate=16000).input_features[0]
    example["labels"] = processor.tokenizer(text_target=example["text"]).input_ids
    return example

# Apply preprocessing
dataset = dataset.map(prepare_example, remove_columns=["path", "text"])

# ----------- Custom Data Collator -------------
@dataclass
class DataCollatorWhisper:
    processor: WhisperProcessor
    padding: str = "longest"

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        batch["labels"] = labels
        return batch

data_collator = DataCollatorWhisper(processor)

# ----------- Training Arguments -------------
training_args = TrainingArguments(
    output_dir="./whisper_finetune_results",
    per_device_train_batch_size=16,  # Try 16 first; adjust up/down if OOM
    gradient_accumulation_steps=1,   # Increase if GPU memory is not enough
    logging_dir="./logs",
    logging_strategy="epoch",        # Log once per epoch
    logging_first_step=True,
    save_strategy="epoch",           # Save checkpoint per epoch
    num_train_epochs=20,             # Increase if you want longer training
    #evaluation_strategy="no",        # or use "epoch" if validation set available
    fp16=True,                       # Use FP16 for faster training on 4090
    learning_rate=2e-5,              # Start slightly higher for Whisper
    lr_scheduler_type="cosine",      # Cosine decay works well
    warmup_steps=500,
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="tensorboard",         # Enables tensorboard logs
)


# ----------- Trainer Setup and Training -------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=processor.tokenizer,
    data_collator=data_collator,
)

trainer.train()

# ----------- Save final model -------------
model.save_pretrained("./whisper_finetuned_model_20")
processor.save_pretrained("./whisper_finetuned_model_20")
