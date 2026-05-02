---
base_model: unsloth/Qwen3-4B-unsloth-bnb-4bit
library_name: transformers
model_name: gamma_0.5
tags:
- generated_from_trainer
- trl
- unsloth
- cpo
licence: license
---

# Model Card for gamma_0.5

This model is a fine-tuned version of [unsloth/Qwen3-4B-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen3-4B-unsloth-bnb-4bit).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with CPO, a method introduced in [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://huggingface.co/papers/2401.08417).

### Framework versions

- TRL: 0.24.0
- Transformers: 5.5.0
- Pytorch: 2.10.0+cu128
- Datasets: 3.6.0
- Tokenizers: 0.22.2

## Citations

Cite CPO as:

```bibtex
@inproceedings{xu2024contrastive,
    title        = {{Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation}},
    author       = {Haoran Xu and Amr Sharaf and Yunmo Chen and Weiting Tan and Lingfeng Shen and Benjamin Van Durme and Kenton Murray and Young Jin Kim},
    year         = 2024,
    booktitle    = {Forty-first International Conference on Machine Learning, {ICML} 2024, Vienna, Austria, July 21-27, 2024},
    publisher    = {OpenReview.net},
    url          = {https://openreview.net/forum?id=51iwkioZpn}
}
```

Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```