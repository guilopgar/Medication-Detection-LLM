# Medication-Detection-LLM

**Medication-Detection-LLM** provides fine-tuned large language models (LLMs) for the automatic extraction of medication mentions from social media texts, specifically Twitter. This repository accompanies our research paper on end-to-end generative models for medication detection, where we achieved state-of-the-art (SOTA) performance in both fine-grained span extraction and coarse-grained tweet-level classification tasks.

Unlike traditional multi-stage NLP pipelines, our models leverage instruction-tuned LLMs to perform direct text generation for medication mention extraction. This approach simplifies the pipeline and improves computational efficiency, while maintaining or surpassing the performance of previous SOTA systems.

---

## 🚀 Models

We release two fine-tuned models based on the Flan-T5 architecture:

* [Flan-T5-Large Medication NER](https://huggingface.co/guilopgar/flan-t5-large-medication-ner)

  * \~800M parameters
  * Best suited for resource-constrained environments
* [Flan-T5-XL Medication NER](https://huggingface.co/guilopgar/flan-t5-xl-medication-ner)

  * 3B parameters
  * Achieves highest performance across tasks

---

## 📊 Results Summary

| Model          | BioCreative VII (Strict F1) | SMM4H'20 F1 | SMM4H'18 F1 |
| -------------- | --------------------------- | ----------- | ----------- |
| Flan-T5-Large  | 80.4% (with lexicon)        | **86.1%**   | 96.2%       |
| Flan-T5-XL     | **80.9% (with lexicon)**    | 84.4%       | **96.7%**   |
| SOTA Baselines | 80.4%                       | 85.4%       | 95.4%       |

* Flan-T5-XL with lexicon filtering achieved the highest F1-score of **80.9%** on the BioCreative VII Shared Task 3 test set, surpassing the official SOTA system.
* On the highly imbalanced SMM4H’20 dataset, Flan-T5-Large achieved the highest F1-score of **86.1%**.
* On the balanced SMM4H’18 dataset, Flan-T5-XL achieved an F1-score of **96.7%**, outperforming all SOTA baselines.

---

## 📥 How to Use

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "guilopgar/flan-t5-large-medication-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Benadryl, bedtime snack, and New Girl. The party is getting real."
prompt = f"You are given a tweet followed by a specific question asking about the content of the tweet. Your objective is to identify and list any drug names, medications, or dietary supplements mentioned in the tweet. If one or more are mentioned, list each distinctly, separated by a comma. If none are mentioned, return an empty list [].\nInput: Tweet: {text}\nQuestion: What are the drugs, medications or dietary supplements mentioned in the tweet?\nOutput:"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📄 Inference Script

To reproduce the results obtained in the paper, as well as to facilitate large-scale analysis of social media data, we also provide a ready-to-use script for batch processing of tweets using our fine-tuned models in the `inference` directory. You can optionally apply lexicon-based filtering to improve precision.

### Usage

```bash
python inference/predict.py \
    --model_name guilopgar/flan-t5-large-medication-ner \
    --input_csv data/sample_tweets.csv \
    --lexicon_path data/lexicon.pkl \
    --output_path data/sample_tweets_predictions.csv
```

* `--model_name`: Hugging Face model name or local path to the fine-tuned model.
* `--input_csv`: Path to a CSV file containing the tweets.
* `--lexicon_path` *(optional)*: Path to a lexicon `.pkl` file for postprocessing (token-based filtering to improve precision).
* `--output_path`: Path to save the predictions CSV file.

### Example Files

* In the `data/` folder, you can find:

  * `sample_tweets.csv`: Example input file containing five tweets.
  * `lexicon.pkl`: Preprocessed drug lexicon used for lexicon-based filtering.

---

## 📚 Citation

If you use these models in your research, please cite our work:

```bibtex
@article{Lopez-Garcia2025.05.16.25327791,
   author = {Lopez-Garcia, Guillermo and Xu, Dongfang and Gonzalez-Hernandez, Graciela},
   title = {Detecting Medication Mentions in Social Media Data Using Large Language Models},
   year = {2025},
   doi = {10.1101/2025.05.16.25327791},
   publisher = {Cold Spring Harbor Laboratory Press},
   URL = {https://www.medrxiv.org/content/early/2025/05/18/2025.05.16.25327791},
   journal = {medRxiv}
}
```
