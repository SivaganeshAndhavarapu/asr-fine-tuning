import os
import torch
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer

# ---------- Configuration ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
csv_path = "/home/hp/sivaganesh/ada_test/train_100.csv"  # test set path
sampling_rate = 16000
base_model_name = "openai/whisper-base.en"
finetuned_model_path = "./whisper_finetuned_model"

# ---------- Load Dataset ----------
dataset = load_dataset("csv", data_files={"test": csv_path}, split="test")
dataset = dataset.cast_column("path", Audio(sampling_rate=sampling_rate))
dataset = dataset.filter(lambda x: x["text"] and x["text"].strip() != "")

# ---------- Load Models and Processor ----------
processor = WhisperProcessor.from_pretrained(base_model_name)

base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name).to(device)
finetuned_model = WhisperForConditionalGeneration.from_pretrained(finetuned_model_path).to(device)

# ---------- Transcription Function ----------
def transcribe(model, audio):
    inputs = processor(audio["array"], sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
    
    # Disable forced language decoding for fine-tuned models
    generation_config = model.generation_config
    generation_config.forced_decoder_ids = None
    
    with torch.no_grad():
        predicted_ids = model.generate(inputs, generation_config=generation_config)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.lower()

# ---------- Evaluation ----------
base_preds = []
finetuned_preds = []
references = []

for example in dataset:
    audio = example["path"]
    ref = example["text"].lower().strip()

    base_transcript = transcribe(base_model, audio)
    finetuned_transcript = transcribe(finetuned_model, audio)

    base_preds.append(base_transcript)
    finetuned_preds.append(finetuned_transcript)
    references.append(ref)

    print(f"REF      : {ref}")
    print(f"BASE     : {base_transcript}")
    print(f"FINETUNED: {finetuned_transcript}")
    print("-" * 50)

# ---------- Compute WER ----------
base_wer = wer(references, base_preds)
finetuned_wer = wer(references, finetuned_preds)

print(f"\nBase Model WER      : {base_wer:.3f}")
print(f"Fine-tuned Model WER: {finetuned_wer:.3f}")