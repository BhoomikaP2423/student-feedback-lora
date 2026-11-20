# Student Feedback Summarizer (LoRA)
VERIFIED [![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/tw47kbncs8/badge.svg)](https://msrit.ask.thili.ai/b/tw47kbncs8)

A small educational project that fine-tunes a pretrained instruction model with LoRA on student feedback and provides a Streamlit interface for concise curriculum-improvement summaries.

## Pipeline
1. Prepare feedback/instructor-summary pairs in `data/feedback.jsonl`.
2. Fine-tune `google/flan-t5-small` with PEFT LoRA using `train.py`.
3. Merge the trained adapter into the base model using `merge_adapter.py`.
4. Launch `app.py`; the UI loads the merged model and summarizes instructor-provided feedback.

> This is a compact demonstration. The included dataset is illustrative/synthetic, not real student data. Replace it with consented, anonymized feedback before making curriculum decisions. Generated summaries should be reviewed by an instructor.

## Google Colab (GPU recommended)
```bash
!unzip -oq student-feedback-lora.zip
%cd student-feedback-lora
!pip install -q -r requirements.txt
!python -m pytest -q
!python train.py --epochs 3
!python merge_adapter.py
```

The trained adapter is saved to `outputs/lora_adapter`; the merged model is saved to `outputs/merged_model`.

To launch Streamlit locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```
In Colab, use a tunnel such as `localtunnel` only if you understand its public-access implications. Do not upload private student feedback to a public tunnel.

## Example input
- Students found recursion difficult and asked for more worked examples.
- The database lab instructions were clear, but the final exercise took too long.
- More quizzes before the midterm would help identify gaps.

## Example output format
- **What is working:** ...
- **Main concerns:** ...
- **Suggested improvements:** ...

## Notes
- Base model: `google/flan-t5-small` (small enough for a teaching demo).
- Fine-tuning: supervised seq2seq training with LoRA adapters via Hugging Face PEFT.
- Merge: `PeftModel.merge_and_unload()` creates a standalone merged model for the Streamlit app.
- Evaluation: training/evaluation loss is logged by Trainer. No benchmark score is claimed.
## Results and Evaluation

The model was fine-tuned using LoRA on 10 illustrative student feedback examples for 3 epochs.

- Training loss: 4.3769
- Trainable parameters: 344,064 (0.4451%)
- Automated tests: 7 passed
- LoRA adapter and merged model: Saved successfully





![Streamlit application](docs/screenshots/app-demo.png)

**Test results**

![Test results](docs/screenshots/tests.png)
