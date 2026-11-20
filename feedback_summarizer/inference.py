import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from feedback_summarizer.prompts import make_prompt

def load_model(model_dir="outputs/merged_model"):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError("Merged model missing. Run train.py and merge_adapter.py first.")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model, device

def summarize(feedback, tokenizer, model, device="cpu", max_new_tokens=90):
    if not feedback or not feedback.strip():
        raise ValueError("Please enter student feedback.")
    prompt = make_prompt(feedback)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                num_beams=4, early_stopping=True)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()
