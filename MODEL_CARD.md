---
library_name: peft
base_model: unsloth/Qwen3-4B-unsloth-bnb-4bit
tags:
- simpo
- alignment
- text-generation-inference
- sales-automation
- judge
---

# Qwen3 4B Tenacious Critic (SimPO)

This is a LoRA-adapted 4-bit critic model developed as part of the **Week 11 Tenacious Sales Agent Evaluation Bench** (Act IV). It was trained using **SimPO** (Simple Preference Optimization) to evaluate and rank B2B sales outreach drafts against the Tenacious verification rubric.

## Intended Use
This model is intended to be deployed as a **rejection-sampling layer** (a "Judge") in front of the Week 10 Conversion Engine composer. 
*   **Input:** A drafted sales email and context.
*   **Output / Reward:** Instead of generating text, it provides a length-normalized token log-probability (SimPO reward) to rank multiple candidates. It penalizes tone-fails, hallucinated signals, and condescending gap-framing.

## Training Configuration
*   **Base Model:** `unsloth/Qwen3-4B-unsloth-bnb-4bit`
*   **Algorithm:** SimPO (pure preference, no NLL mixing)
*   **LoRA Rank:** 16
*   **LoRA Alpha:** 32
*   **Beta (Reward scale):** 2.0
*   **Gamma (Margin):** 0.5
*   **Precision:** fp16 + 4-bit QLoRA
*   **Infrastructure:** Google Colab T4 (16 GB VRAM) leveraging Unsloth

## Evaluation Metrics (Tenacious-Bench v0.1 Dev Partition)
During ablation, this specific `gamma=0.5` checkpoint achieved the following zero-shot metrics on the held-out development partition:

*   **Preference Accuracy:** 1.0 (100%)
*   **Average Reward Gap:** 1.333
*   **Judge-Evaluator Agreement:** 1.0 (100% agreement with the deterministic [scoring_evaluator.py](cci:7://file:///home/kg/Projects/10Academy/sales-agent-evaluation-bench/scoring_evaluator.py:0:0-0:0))

*Prior to training, the baseline `Qwen3-4B` model had a preference accuracy of merely 8.65% with a negative reward gap.*

## How to Load
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_id = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
adapter_id = "kgutd/Qwen3-4B-Tenacious-Critic-SimPO"

# Load the base model
model = AutoModelForCausalLM.from_pretrained(base_model_id, load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Apply the trained LoRA adapter
model = PeftModel.from_pretrained(model, adapter_id)
