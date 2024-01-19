# interlocking-DUS(iDUS) is modification of Depth-Up Scaling(DUS) introduced by SOLAR-10.7B
# iDUS does not simply attach model layers, but builds them by interlocking them
#
# This code refers to silphendio's gist(https://gist.github.com/silphendio/90f7e23b2b1ab6949fd4b35e7dd705cf)
# Gist: https://gist.github.com/gauss5930/f8821422825feff09bc0872343a0ee59

from transformers import  AutoTokenizer, AutoModelForCausalLM, AutoTokenizer
import huggingface_hub
import torch
import argparse

def args_parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("--DUS_type", type=str, required=True, help="You can choose original, interlocked_1 or interlocked_8")

    parser.add_argument("--init_model_path", type=str, help="The path of model that Llama2 initialized with Mistral")

    parser.add_argument("--hf_token", type=str)
    parser.add_argument("--save_type", type=str, help="You can choose hf or directory")
    parser.add_argument("--save_path", type=str)

    return parser.parse_args()

def main():
    args = args_parse()

    if not args.save_type in ["hf", "directory"]:
        raise ValueError("You should choose save_type between hf and directory!!")

    if not args.DUS_type in ["original", "interlocked_1", "interlocked_8"]:
        raise ValueError("You should choose save_type between original, interlocked_1 and interlocked_8!!")

    model_path = args.init_model_path # huggingface name or local folder
    new_model_architecture = 'upstage/SOLAR-10.7B-v1.0' # Same size model

    if args.DUS_type == "original":
        layer_arrangement = list(range(0, 24)) + list(range(8, 32))
    elif args.DUS_type == "interlocked_1":
        # iDUS-1layer merges each layer by interlocking with each other
        layer_arrangement = []
        layer_A, layer_B = list(range(0, 24)), list(range(8, 32))
        for i in range(len(layer_A)):
            layer_arrangement.append(layer_A[i])
            layer_arrangement.append(layer_B[i])
    else:
        # iDUS-8layer divides layers into groups, and merges them by interlocking with each other
        layer_arrangement = list(range(0, 8)) + list(range(8, 16)) * 2 + list(range(16, 24)) * 2 + list(range(24, 32))

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # rearrange layers
    new_state_dict = model.state_dict().copy()
    layer_keys_template = [key.replace('.0.', '.{}.') for key in model.state_dict() if '.0.' in key]

    for new_layer, old_layer in enumerate(layer_arrangement):
        for key in layer_keys_template:
            new_state_dict[key.format(new_layer)] = model.state_dict()[key.format(old_layer)]

    new_config = model.config
    new_config.n_layer = len(layer_arrangement)
    new_config.num_hidden_layers = len(layer_arrangement)

    # load the merged model and tokenizer
    new_model = AutoModelForCausalLM.from_pretrained(new_model_architecture, torch_dtype=torch.float16, config=new_config, state_dict=new_state_dict)

    # Upload or save the model and tokenizer
    if args.save_type == "directory":
        new_model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
    else:
        new_model.push_to_hub(args.save_path)
        tokenizer.push_to_hub(args.save_path)

if __name__ == "__main__":
    main()