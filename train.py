"""
Continued pretraining (unsupervised causal language modeling) of an LLM on raw text.

The model learns to predict the next token on your text, exactly like pretraining.
No labels, no prompt/answer pairs: just plain text.

Defaults are set up for a GPU server:
  model   = Qwen/Qwen3.8-27B            (loaded text-only; the vision tower is dropped)
  dataset = tascib/turkish-llm-dataset  (Turkish pretraining corpus, `text` column)
  method  = LoRA, bf16, gradient checkpointing, 2048-token packed blocks

Examples:
  # 1) Single GPU (needs ~80 GB, e.g. A100/H100 80GB), all defaults
  uv run train.py

  # 2) All GPUs on the node (DDP, every GPU holds a bf16 copy of the model)
  uv run torchrun --nproc_per_node=gpu train.py

  # 3) Full fine-tune (all weights), sharded with FSDP. Needs ~8x80 GB GPUs.
  uv run torchrun --nproc_per_node=gpu train.py --full_finetune --fsdp

  # 4) Use more of the corpus (each shard is ~0.16 GB / ~220k documents; 0 = all shards)
  uv run torchrun --nproc_per_node=gpu train.py --hf_num_files 50

  # 5) Your own .txt files instead of the Hub dataset
  uv run train.py --train_files "data/*.txt"
"""

import argparse
import math
import os
from itertools import chain
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # we parallelize with num_proc instead

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_DATASET = "tascib/turkish-llm-dataset"
# Downloaded models go to <storage>/models, datasets (raw + tokenized cache) to <storage>/datasets
DEFAULT_STORAGE = Path(__file__).resolve().parent / "storage"


def parse_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--model_name", default=DEFAULT_MODEL,
                   help="Any causal LM on the Hugging Face Hub (base models work best).")
    p.add_argument("--storage_dir", default=str(DEFAULT_STORAGE),
                   help="Where models/ and datasets/ are downloaded. Put this on a large disk.")
    p.add_argument("--attn_implementation", default="sdpa",
                   help="sdpa (default) or flash_attention_2 if flash-attn is installed.")
    # Data
    p.add_argument("--hf_dataset", default=DEFAULT_DATASET)
    p.add_argument("--hf_config", default=None)
    p.add_argument("--hf_num_files", type=int, default=10,
                   help="How many data/dataset_part_N.jsonl shards to use (~0.16 GB each). 0 = all.")
    p.add_argument("--train_files", default=None,
                   help="Glob for local .txt files. If set, overrides --hf_dataset.")
    p.add_argument("--text_column", default="text")
    p.add_argument("--max_samples", type=int, default=None, help="Limit number of documents.")
    p.add_argument("--block_size", type=int, default=2048, help="Tokens per training sequence.")
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--max_eval_docs", type=int, default=2000,
                   help="Cap on validation documents so evaluation stays fast on large corpora.")
    p.add_argument("--num_proc", type=int, default=min(32, os.cpu_count() or 1),
                   help="CPU processes for tokenization/packing.")
    # Training
    p.add_argument("--output_dir", default="out/cpt-model")
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=None, help="Default: 1e-4 with LoRA, 1e-5 full fine-tune.")
    p.add_argument("--batch_size", type=int, default=2, help="Per GPU.")
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--no_gradient_checkpointing", action="store_true",
                   help="Faster but uses much more memory.")
    p.add_argument("--full_finetune", action="store_true",
                   help="Train all weights instead of LoRA adapters (use with --fsdp for 27B).")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--fsdp", action="store_true", help="Shard model across GPUs (launch with torchrun).")
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--skip_initial_eval", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.lr is None:
        args.lr = 1e-5 if args.full_finetune else 1e-4
    args.model_cache = os.path.join(args.storage_dir, "models")
    args.data_cache = os.path.join(args.storage_dir, "datasets")
    os.makedirs(args.model_cache, exist_ok=True)
    os.makedirs(args.data_cache, exist_ok=True)
    return args


def load_raw_dataset(args):
    if args.train_files:
        # "paragraph" = each blank-line-separated block becomes one document
        ds = load_dataset("text", data_files={"train": args.train_files},
                          sample_by="paragraph", split="train", cache_dir=args.data_cache)
    else:
        # Download raw files into <storage>/datasets/raw/<repo> (not the global HF cache)
        from huggingface_hub import snapshot_download
        raw_dir = os.path.join(args.data_cache, "raw", args.hf_dataset.replace("/", "__"))
        if args.hf_dataset == DEFAULT_DATASET and args.hf_num_files > 0:
            # Only download the first N shards instead of the full ~158 GB corpus
            files = [f"data/dataset_part_{i}.jsonl" for i in range(args.hf_num_files)]
            snapshot_download(args.hf_dataset, repo_type="dataset", local_dir=raw_dir,
                              allow_patterns=files, max_workers=16)
            ds = load_dataset("json", data_files={"train": [os.path.join(raw_dir, f) for f in files]},
                              split="train", cache_dir=args.data_cache)
        else:
            snapshot_download(args.hf_dataset, repo_type="dataset", local_dir=raw_dir, max_workers=16)
            ds = load_dataset(raw_dir, args.hf_config, split="train", cache_dir=args.data_cache)

    ds = ds.filter(lambda x: x[args.text_column] is not None and len(x[args.text_column].strip()) > 0,
                   num_proc=args.num_proc)
    if args.max_samples:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.max_samples, len(ds))))
    n_val = max(1, min(math.ceil(len(ds) * args.val_ratio), args.max_eval_docs))
    return ds.train_test_split(test_size=n_val, seed=args.seed)


def doc_separator_id(tokenizer):
    # Instruct/chat models use <|im_end|> as EOS; for raw documents <|endoftext|> is the right boundary.
    tok_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if tok_id is not None and tok_id != tokenizer.unk_token_id:
        return tok_id
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has no EOS token; pick a different model.")
    return tokenizer.eos_token_id


def build_lm_dataset(raw, tokenizer, args):
    sep = doc_separator_id(tokenizer)

    # Step A: tokenize every document and add an end-of-text token between documents
    def tokenize(batch):
        ids = tokenizer(batch[args.text_column], add_special_tokens=False)["input_ids"]
        return {"input_ids": [x + [sep] for x in ids]}

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names,
                        num_proc=args.num_proc, desc="Tokenizing")

    # Step B: concatenate all tokens and cut them into fixed-size blocks ("packing").
    # labels = input_ids; the model shifts them internally to predict the next token.
    bs = args.block_size

    def group(batch):
        all_ids = list(chain(*batch["input_ids"]))
        total = (len(all_ids) // bs) * bs
        blocks = [all_ids[i:i + bs] for i in range(0, total, bs)]
        return {"input_ids": blocks,
                "attention_mask": [[1] * bs for _ in blocks],
                "labels": [b.copy() for b in blocks]}

    return tokenized.map(group, batched=True, batch_size=1000, num_proc=args.num_proc,
                         desc=f"Packing into {bs}-token blocks")


def main():
    args = parse_args()
    set_seed(args.seed)

    on_gpu = torch.cuda.is_available()
    use_bf16 = on_gpu and torch.cuda.is_bf16_supported()
    use_fp16 = on_gpu and not use_bf16
    if on_gpu:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.model_cache)

    # Model class name for FSDP wrapping is read from the config without loading weights
    fsdp_kwargs = {}
    if args.fsdp:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_name, cache_dir=args.model_cache)
        decoder_layer = "Qwen3_5DecoderLayer" if "qwen3_5" in cfg.model_type else None
        fsdp_kwargs = dict(
            fsdp="full_shard auto_wrap",
            fsdp_config={
                **({"transformer_layer_cls_to_wrap": [decoder_layer]} if decoder_layer else {}),
                "activation_checkpointing": not args.no_gradient_checkpointing,
                "cpu_ram_efficient_loading": True,
                "sync_module_states": True,
                "use_orig_params": True,
            },
        )

    # TrainingArguments must exist before the model is loaded (sets up distributed/FSDP state)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # float < 1 = ratio of total steps
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch_fused" if on_gpu else "adamw_torch",
        # With FSDP, checkpointing is done by FSDP itself (activation_checkpointing above)
        gradient_checkpointing=not args.no_gradient_checkpointing and not args.fsdp,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=on_gpu,
        ddp_find_unused_parameters=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=not args.fsdp,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=args.seed,
        **fsdp_kwargs,
    )

    # Only rank 0 downloads/tokenizes; other ranks then read the cached result
    with training_args.main_process_first(desc="dataset preparation"):
        raw = load_raw_dataset(args)
        lm_ds = build_lm_dataset(raw, tokenizer, args)
    n_train, n_val = len(lm_ds["train"]), len(lm_ds["test"])
    print(f"Train blocks: {n_train} | Val blocks: {n_val} | ~{n_train * args.block_size:,} train tokens")
    if n_train == 0 or n_val == 0:
        raise ValueError("Not enough text for even one block. Add data or lower --block_size.")

    # LoRA: frozen base weights in bf16 (54 GB for 27B). Full fine-tune: fp32 master weights,
    # Trainer runs bf16 autocast (bf16 weights would lose small updates).
    load_dtype = torch.float32 if (args.full_finetune or not on_gpu) else (
        torch.bfloat16 if use_bf16 else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=args.model_cache,
        dtype=load_dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False

    if not args.full_finetune:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                              target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds["test"],
        data_collator=default_data_collator,
    )

    # Auto-resume if the job was interrupted (common on shared servers)
    last_ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last_ckpt:
        print(f"Resuming from {last_ckpt}")
    elif not args.skip_initial_eval:
        before = trainer.evaluate()
        print(f"Perplexity BEFORE training: {math.exp(before['eval_loss']):.2f}")

    trainer.train(resume_from_checkpoint=last_ckpt)

    after = trainer.evaluate()
    print(f"Perplexity AFTER training:  {math.exp(after['eval_loss']):.2f}")

    final_dir = os.path.join(args.output_dir, "final")
    if args.fsdp or args.full_finetune:
        # Trainer gathers sharded weights correctly (adapter only if LoRA+FSDP)
        trainer.save_model(final_dir)
    elif trainer.is_world_process_zero():
        merged = model.merge_and_unload()  # bake adapters into the base weights
        merged.save_pretrained(final_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final_dir)
        print(f"Saved model to {final_dir}")


if __name__ == "__main__":
    main()
