# This code initialize Llama2-7B model with weigths of Mistral-7B as was done in SOLAR-10.7B
#
# Gist: https://gist.github.com/gauss5930/0e6206bedff2f049e826e0b6c6957b51

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import huggingface_hub
import torch
import argparse

def args_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--save_type", type=str, help="You can choose hf or directory")
    parser.add_argument("--save_path", type=str)

    return parser.parse_args()

def main():
    args = args_parse()

    if not args.save_type in ["hf", "directory"]:
        raise ValueError("You should choose save_type between hf and directory!!")
    
    huggingface_hub.login(args.hf_token)
    
    # Load the configuration of Llama2 with modification on hidden_dim & num_kv_heads equal to those of Mistral
    model_config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf", token=args.hf_token, intermediate_size=14336, num_key_value_heads=8, torch_dtype="float16")
    # Load the model with Llama2 architecture and Mistral weights
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", tordh_dtype=torch.float16, config=model_config)
    # Load the tokenizer of Mistral-7B
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    if args.save_type == "directory":
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
    else:
        model.push_to_hub(args.save_path)
        tokenizer.push_to_hub(args.save_path)

if __name__ == "__main__":
    main()