from argparse import ArgumentParser
import torch
from models import GPT
from tokenizer import TiktokenTokenizer

device = "cuda:0" if torch.cuda.is_available else "cpu"

def load_model(model, args):
     checkpoint = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
     print(checkpoint)
     checkpoint = {name.split("actor._orig_mod.model.")[1]: param for name, param in checkpoint.items() if "actor._orig_mod.model." in name}
     model.load_state_dict(checkpoint, strict=True)
     return model

@torch.no_grad()
def predict(prompt, model, tokenizer, args):
    tokens = tokenizer(prompt,
                       max_length=args.sequence_length,
                       truncation=True,
                       return_tensors="pt")
    
    idx = tokens['input_ids'].unsqueeze(0).to(device)
    completions = model.generate(idx, args.sequence_length, 0.4)

    text = tokenizer.enc.decode(completions[0].cpu().tolist())
    return text

def main(args):
    
    tokenizer = TiktokenTokenizer("gpt2")

    model = GPT(args.lora_rank,
                args.vocab_size,
                args.n_heads,
                args.embedding_dim,
                args.n_layers,
                1024,
                args.dropout_rate,
                args.activation_checkpointing,
                args.use_bias)
    
    model = load_model(model, args).to(device)
    model = torch.compile(model)
    model.eval()

    while True:
        print("="*20+"\n")
        input_str = input("Human:")+"\n"
        print("Assistant: ", predict(input_str, model, tokenizer, args))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--lora_rank', type=int, default=0)
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--embedding_dim', default=1024, type=int)

    parser.add_argument('--n_layers', type=int, default=24)

    parser.add_argument('--sequence_length', type=int, default=256)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--activation_checkpointing', type=bool, default=False)
    parser.add_argument('--use_bias', type=bool, default=True)
    parser.add_argument('--checkpoint', type=str, default="./models/PPO/ppo.ckpt")

    args = parser.parse_args()
    main(args)