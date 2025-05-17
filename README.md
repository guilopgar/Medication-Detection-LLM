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
* On the balanced SMM4H’18 dataset, Flan-T5-XL achieved an F1-score of **96.7%**, outperforming all baselines.

---

## 📚 Citation

If you use these models in your research, please cite our work:

```bibtex
@misc{MEDRXIV/2025/327791,
  author = {Lopez-Garcia, Guillermo and Xu, Dongfang and Gonzalez-Hernandez, Graciela},
  title = {Detecting Medication Mentions in Social Media Data Using Large Language Models},
  publisher = {medRxiv},
  year = {2025}
}
```

---

## 📥 How to Use

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "guilopgar/flan-t5-large-medication-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "Bedtime snack, benadryl, New Girl, and acetaminophen. The party is getting real."
prompt = f"You are given a tweet followed by a specific question asking about the content of the tweet. Your objective is to identify and list any drug names, medications, or dietary supplements mentioned in the tweet. If one or more are mentioned, list each distinctly, separated by a comma. If none are mentioned, return an empty list [].\nInput: Tweet: {text}\nQuestion: What are the drugs, medications or dietary supplements mentioned in the tweet?\nOutput:"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📧 Contact

For questions or feedback, feel free to reach out via [GitHub Issues](https://github.com/guilopgar/Medication-Detection-LLM/issues) or contact the authors directly.
