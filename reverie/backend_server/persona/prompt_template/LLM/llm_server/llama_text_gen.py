import fire
from llama import Llama
from typing import List, Union






def generate_text(
    prompt: Union[str, List[str]],
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Generates text based on the provided prompt(s) using a pretrained model.

    Args:
        prompt (Union[str, List[str]]): The prompt(s) for text generation.
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
        max_seq_len (int, optional): The maximum sequence length for input prompts.
        max_gen_len (int, optional): The maximum length of generated sequences.
        max_batch_size (int, optional): The maximum batch size for generating sequences.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    if isinstance(prompt, str):
        prompts = [prompt]
    else:
        prompts = prompt

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")

def main(
    model_type: str,  # Replace ckpt_dir and tokenizer_path with model_type
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    prompt: Union[str, List[str]] = None,
):
    if model_type not in model_paths:
        print(f"Model type '{model_type}' not recognized. Please provide a valid model type.")
        return

    # Retrieve model paths based on the selected model type
    ckpt_dir = model_paths[model_type]["ckpt_dir"]
    tokenizer_path = model_paths[model_type]["tokenizer_path"]

    if prompt:
        generate_text(
            prompt,
            ckpt_dir,
            tokenizer_path,
            temperature,
            top_p,
            max_seq_len,
            max_gen_len,
            max_batch_size,
        )
    else:
        print("Please provide a prompt for text generation.")

if __name__ == "__main__":
    fire.Fire(main)