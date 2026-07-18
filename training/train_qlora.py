"""QLoRA fine-tune of Qwen2.5-7B-Instruct on the Cyber-Witten distillation set.

Runs on a single rented GPU (RTX 4090 24GB works; A100 80GB is comfortable).
Self-contained: standard trl + peft + bitsandbytes, no framework config files.
Loss is computed on ASSISTANT tokens only (the answer), not on the prompt.

    python train_qlora.py --train sft_train.jsonl --val sft_val.jsonl \
        --out ./cyber-witten-lora

See training/README.md for the full rent-a-GPU runbook (install, run, merge,
GGUF-quantize, deploy to ollama).
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct",
                    help="Must match the serving model family (ollama qwen2.5:7b is the instruct variant)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="./cyber-witten-lora")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset

    def _pick_attn() -> str:
        # flash-attn wheels are often unavailable on rented boxes (esp. domestic
        # Chinese platforms); sdpa is ~10-20% slower but always works.
        if torch.cuda.is_available():
            try:
                import flash_attn  # noqa: F401
                return "flash_attention_2"
            except ImportError:
                print("flash-attn not installed; falling back to sdpa")
        return "sdpa"

    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base)

    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
        attn_implementation=_pick_attn(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    data = load_dataset("json", data_files={"train": args.train, "val": args.val})

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        bf16=True,
        max_length=args.max_seq_len,
        packing=False,                    # long structured samples; do not pack
        assistant_only_loss=True,         # loss on the answer, not the prompt
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        save_strategy="steps",     # checkpoint often: rented hosts can die mid-run
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=20260717,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=cfg,
        train_dataset=data["train"],
        eval_dataset=data["val"],
        peft_config=lora,
    )
    import glob
    has_ckpt = bool(glob.glob(f"{args.out}/checkpoint-*"))
    if has_ckpt:
        print("resuming from last checkpoint")
    trainer.train(resume_from_checkpoint=has_ckpt)
    trainer.save_model(args.out)
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
