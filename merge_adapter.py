import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "google/flan-t5-small"
ADAPTER_DIR = "outputs/lora_adapter"
MERGED_DIR = "outputs/merged_model"

def main():
    if not os.path.isdir(ADAPTER_DIR):
        raise FileNotFoundError("Adapter not found. Run train.py first.")
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    merged = model.merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR)
    AutoTokenizer.from_pretrained(ADAPTER_DIR).save_pretrained(MERGED_DIR)
    print(f"Merged model saved to {MERGED_DIR}")

if __name__ == "__main__":
    main()
