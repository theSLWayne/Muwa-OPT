# Muwa-OPT - A budget-friendly OPT-based LLM 

![](Muwa.png)

Muwa is a fine-tuned LoRA model based on Facebook's OPT model architecture. Muwa was fine-tuned using the [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), which is a dataset of instruction-following records that belong to multiple categories like brainstorming, classification, closed QA, generation, information extraction, open QA, and summarization. **The specialty of Muwa is that only free resources have been used to fine-tune the model**, no fancy arrays of GPUs or paid GPU processors were not used for fine-tuning the model; only the free-tier of Google Colaboratory.

Muwa is currently trained using the [OPT 1.3b model](https://huggingface.co/facebook/opt-1.3b), which is available in HuggingFace. 

This work is heavily inspired from [Yudhanjaya's Eluwa model](https://github.com/yudhanjaya/Eluwa). Most of the model fine-tuning and benchmarking code is taken from their repository and I made some adjustments to the code and changed some parameters to make sure that the fine-tuning process can be done on free resources that were available to me at the time.

## Inference

Make sure you install the following Python packages in the environment where the model is intended to be run.

```shell
pip install torch peft datasets evaluate transformers accelerate bitsandbytes
```

First, OPT 1.3b model should be loaded and then Muwa should be loaded from its HuggingFace repository. After the models are loaded, they can be used for inference.

```python
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Define model names to be loaded
peft_model_id = 'theSLWayne/Muwa-1.3b'
base_model = 'facebook/opt-1.3b'

# Load base model
model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map='auto',
            torch_dtype=torch.float16,
        )

# Load Muwa
model = PeftModel.from_pretrained(
            model,
            peft_model_id,
            device_map='auto',
            torch_dtype=torch.float16,
        )

# Initiate tokenizer of the base model
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Create batches of inputs
batch = tokenizer("What is a deep learning model?", return_tensors='pt')

# Take predictions
with torch.cuda.amp.autocast():
  output_tokens = model.generate(**batch, max_new_tokens=50)

print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))
```

If you intend to use CPU (which is not recommended), you can load the models as follows:

```python
model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map='auto', low_cpu_mem_usage=True
        )

model = PeftModel.from_pretrained(
            model,
            peft_model_id,
            device_map='auto',
        )
```

## Training Muwa



## Evaluating Muwa

## 