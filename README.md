# Muwa-OPT - A budget-friendly OPT-based LLM 

![Muwa Cover Image](Muwa.png)

Muwa is a fine-tuned LoRA model based on Facebook's OPT model architecture. Muwa was fine-tuned using the [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), which is a dataset of instruction-following records that belong to multiple categories like brainstorming, classification, closed QA, generation, information extraction, open QA, and summarization. **The specialty of Muwa is that only free resources have been used to fine-tune the model**, no fancy arrays of GPUs or paid GPU processors were not used for fine-tuning the model; only the free-tier of Google Colaboratory.

Muwa is currently trained using the [OPT 1.3b model](https://huggingface.co/facebook/opt-1.3b), which is available in HuggingFace. 

This work is heavily inspired from [Yudhanjaya's Eluwa model](https://github.com/yudhanjaya/Eluwa). Most of the model fine-tuning and benchmarking code is taken from their repository and I made some adjustments to the code and changed some parameters to make sure that the fine-tuning process can be done on free resources that were available to me at the time.

## Inference

Make sure you install the following Python packages in the environment where the model is intended to be run.

```shell
pip install torch peft datasets evaluate transformers accelerate bitsandbytes
```

First, OPT 1.3b model should be loaded and then Muwa should be loaded from their respective HuggingFace repositories. After the models are loaded, they can be used for inference.

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

This model was fine-tuned for 2 Epochs using the aforementioned Databricks Dolly 15K dataset. This model and its base model (OPT 1.3b) can be loaded in 8-bit. The notebook that was used for training this model can be found on this repo, including my notes on each code block.

The model was trained only using T4 GPU provided by Google Colab. **In order to fit the whole model and the dataset into it, the dataset had an input limit of 1024 tokens per each query**. **This was done because with the default value, the GPU RAM was not enough to fine-tune the model**.

With the limit in input tokens, the model training took ~12 GB of GPU RAM.

### LoRA and PEFT

TODO: Mention the LoRA paper, briefly how it works and the [youtube link](https://www.youtube.com/watch?v=_K3HgjnRHCY&lc=Ugyqpr8yVUW2DHlvsoZ4AaABAg) of paper explanation video.

## Testing and Evaluating

Muwa was tested and evaluated using SQuAD mini, wikitext, and piqa datasets. Both Muwa and its base model, OPT 1.3b were evaluated seperately using all mentioned datasets and the results can be summarized as follows:

| Dataset | OPT 1.3b | Muwa |
|---------|----------|------|
| SQuAD Mini (*avg. f1 score*) | 24.587 | **26.234** |
| wikitext (*perplexity*) | 13.91406 | **13.96875** |
| piqa (*accuracy*) | 0.495 | **0.532** |

As shown, Muwa has been able to outperform its base model by fine tuning using a rather smaller dataset (compared to others like [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) available for these tasks) for all the evaluation datasets. 

This shows that LLMs that have Billions of parameters can be fine-tuned using resources which are available for free and you can actually improve the model's performance by doing so.

Code used for evaluating Muwa can be found in the notebook which is included in this repo.

## The Story Behind Muwa


## License

