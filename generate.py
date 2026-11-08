"""
Test the trained model by letting it continue a piece of text.

  python generate.py --model out/cpt-model/final --prompt "The old lighthouse keeper"
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="out/cpt-model/final")
p.add_argument("--prompt", default="The old lighthouse keeper")
p.add_argument("--max_new_tokens", type=int, default=150)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

inputs = tok(args.prompt, return_tensors="pt").to(device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                         temperature=0.8, top_p=0.95, repetition_penalty=1.1,
                         pad_token_id=tok.eos_token_id)
print(tok.decode(out[0], skip_special_tokens=True))
