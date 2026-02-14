import modal
from modal import App, Image

# Setup

# Create a Modal App named "pricer". This serves as a container for our functions.
app = modal.App("pricer")
# Define a container image using Debian Slim as the base.
# Install necessary Python libraries for deep learning, transformers, quantization, and fine-tuning.
image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate", "peft")
# Define the secrets to be used by the app, likely for Hugging Face API keys.
secrets = [modal.Secret.from_name("hf-secret")]

# Constants

# Specify the GPU to be used for the function.
GPU = "T4"
# The base model to be used for fine-tuning.
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
# The name of the project.
PROJECT_NAME = "pricer"
# The Hugging Face username where the fine-tuned model is stored.
HF_USER = "ed-donner" # your HF name here! Or use mine if you just want to reproduce my results.
# The specific run name of the fine-tuning experiment.
RUN_NAME = "2024-09-13_13.04.39"
# The combined project and run name.
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
# The specific revision (commit hash) of the fine-tuned model.
REVISION = "e8d637df551603dc86cd7a1598a8f44af4d7ae36"
# The full name of the fine-tuned model on the Hugging Face model hub.
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"


@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def price(description: str) -> float:
    """
    This Modal function estimates the price of a product using a fine-tuned language model.
    It loads a base model, applies a PEFT (Parameter-Efficient Fine-Tuning) adapter to it,
    and then uses the fine-tuned model to generate a price based on the product description.

    Args:
        description: The description of the product to be priced.

    Returns:
        The estimated price of the product as a float.
    """
    import os
    import re
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
    from peft import PeftModel

    # The question and prefix used to construct the prompt for the model.
    QUESTION = "How much does this cost to the nearest dollar?"
    PREFIX = "Price is $"

    # Construct the full prompt.
    prompt = f"{QUESTION}\n{description}\n{PREFIX}"
    
    # Quantization configuration to reduce model size and memory usage.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load the tokenizer and the base model.
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, 
        quantization_config=quant_config,
        device_map="auto"
    )

    # Apply the PEFT adapter to the base model to get the fine-tuned model.
    fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL, revision=REVISION)

    # Set a seed for reproducibility.
    set_seed(42)
    # Encode the prompt and generate the output.
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    attention_mask = torch.ones(inputs.shape, device="cuda")
    outputs = fine_tuned_model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1)
    result = tokenizer.decode(outputs[0])

    # Extract the price from the generated text.
    contents = result.split("Price is $")[1]
    contents = contents.replace(',','')
    match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
    return float(match.group()) if match else 0
