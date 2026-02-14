import modal
from modal import App, Volume, Image

# Setup

# Create a Modal App named "llama". This serves as a container for our functions.
app = modal.App("llama")

# Define a container image using Debian Slim as the base.
# Install necessary Python libraries: torch for deep learning,
# transformers for the LLaMA model, bitsandbytes for quantization,
# and accelerate to optimize PyTorch execution.
image = Image.debian_slim().pip_install("torch", "transformers", "bitsandbytes", "accelerate")

# Define the secrets to be used by the app. This is likely for API keys,
# in this case, a Hugging Face secret named "hf-secret".
secrets = [modal.Secret.from_name("hf-secret")]

# Specify the GPU to be used for the function. "T4" is a specific type of NVIDIA GPU.
GPU = "T4"

# Define the name of the pre-trained model to be used from the Hugging Face model hub.
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B" # "google/gemma-2-2b"



@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def generate(prompt: str) -> str:
    """
    This function generates text using a pre-trained LLaMA model.
    It is decorated with @app.function to be run on Modal's infrastructure.

    Args:
        prompt: The input text to generate from.

    Returns:
        The generated text as a string.
    """
    import os
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

    # Quantization configuration to reduce model size and memory usage.
    # load_in_4bit=True: Load the model in 4-bit precision.
    # bnb_4bit_use_double_quant=True: Use double quantization for more memory savings.
    # bnb_4bit_compute_dtype=torch.bfloat16: Use bfloat16 for computation.
    # bnb_4bit_quant_type="nf4": Use the "nf4" quantization type.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load the tokenizer and model from the Hugging Face model hub.
    
    # Load the tokenizer for the specified model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Set the padding token to be the end-of-sentence token.
    tokenizer.pad_token = tokenizer.eos_token
    # Set the padding side to the right.
    tokenizer.padding_side = "right"
    
    # Load the pre-trained causal language model with the specified quantization config.
    # device_map="auto" automatically distributes the model across available devices (e.g., GPU).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        quantization_config=quant_config,
        device_map="auto"
    )

    # Set a seed for reproducibility.
    set_seed(42)
    # Encode the input prompt into tensors and move them to the GPU.
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    # Create an attention mask to indicate which tokens should be attended to.
    attention_mask = torch.ones(inputs.shape, device="cuda")
    # Generate text using the model.
    # max_new_tokens=5: Generate a maximum of 5 new tokens.
    # num_return_sequences=1: Generate only one sequence.
    outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=5, num_return_sequences=1)
    # Decode the generated tokens back into a string.
    return tokenizer.decode(outputs[0])
