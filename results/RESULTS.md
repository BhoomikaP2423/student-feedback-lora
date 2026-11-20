# Student Feedback Summarizer – Results and Evaluation

## 1. Training Results

The project uses FLAN-T5-small as the base model and LoRA for parameter-efficient fine-tuning.

| Metric | Result |
|---|---:|
| Training examples | 10 |
| Epochs | 3 |
| Training steps | 9 |
| Training loss | 4.3769 |
| Total parameters | 77,305,216 |
| Trainable parameters | 344,064 |
| Trainable parameter percentage | 0.4451% |

The LoRA adapter was saved successfully, and the adapter merging process completed successfully.

## 2. Model Evaluation

The model is intended to summarize student feedback into concise, readable summaries.

Evaluation should consider:

- Relevance: Does the summary represent the original feedback?
- Coverage: Does it capture important positive and negative points?
- Clarity: Is the generated summary easy to understand?
- Conciseness: Does it avoid unnecessary repetition?
- Consistency: Does it preserve the meaning of the feedback?

## 3. Baseline Comparison

The base FLAN-T5-small model is used as the baseline. It is compared with the LoRA fine-tuned model using the same feedback inputs.

| Evaluation aspect | Base FLAN-T5-small | LoRA fine-tuned model |
|---|---|---|
| Model | Original pretrained model | LoRA adapted model |
| Fine-tuning | No | Yes |
| Feedback input | Same test inputs | Same test inputs |
| Summary output | To be evaluated | To be evaluated |
| Relevance | To be evaluated | To be evaluated |
| Clarity | To be evaluated | To be evaluated |

Actual generated summaries and evaluation observations should be added after running both models.

## 4. Testing Results

- Automated tests: 7 passed.
- LoRA adapter creation: Successful.
- Model merging: Successful.
- Streamlit application: Started successfully.

## 5. Limitations

- The current training dataset contains only 10 illustrative examples.
- The training loss alone does not establish summary quality.
- More diverse student feedback is needed for reliable evaluation.
- A larger independent test dataset is required for meaningful quantitative metrics.

## 6. Conclusion

The project demonstrates parameter-efficient fine-tuning of FLAN-T5-small using LoRA for student feedback summarization. Only 0.4451% of the model parameters were trainable during fine-tuning. The adapter was saved and merged successfully, and the application was launched. Further evaluation with a larger test dataset is needed to measure summarization quality reliably.
