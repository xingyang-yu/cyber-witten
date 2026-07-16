# Training runbook — distill the teacher into a local qwen2.5-7b

End to end: rent a GPU, QLoRA-train on the distillation set, merge, quantize
to GGUF, serve with ollama, and measure against the base model with the eval
harness. Expected cost: a few dollars of GPU time on top of the ~$12 of
teacher data.

## 0. Prepare data (local, free)

```bash
.venv/bin/python -m scripts.distill_export \
    data/distill/samples_pilot.jsonl data/distill/samples_v1.jsonl
# -> data/distill/sft_train.jsonl, sft_val.jsonl (+ length stats; keep p99 <= max-seq-len)
```

## 1. Rent a GPU

RunPod / Vast.ai / Lambda. Either works:

| GPU | VRAM | ~price | 2k-sample 2-epoch wall time |
|---|---|---|---|
| RTX 4090 | 24 GB | ~$0.4/hr | ~2-3 h (batch 1, grad-accum 8) |
| A100 80GB | 80 GB | ~$1.5/hr | ~1 h (can raise batch to 4) |

Pick a PyTorch 2.x + CUDA 12 image, ≥60 GB disk.

## 2. Set up + upload

```bash
pip install "transformers>=4.46" "trl>=0.13" peft bitsandbytes datasets accelerate flash-attn --no-build-isolation
# then copy up: training/train_qlora.py, data/distill/sft_train.jsonl, sft_val.jsonl
# (runpodctl send / scp / croc — anything)
```

## 3. Train

```bash
python train_qlora.py --train sft_train.jsonl --val sft_val.jsonl --out ./cyber-witten-lora
# watch: train loss falling, eval loss not rising (2 epochs is the default; stop early if val turns)
```

## 4. Merge + quantize to GGUF (still on the GPU box — the Air shouldn't chew 15GB fp16)

```bash
python - <<'EOF'
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16)
m = PeftModel.from_pretrained(base, "./cyber-witten-lora").merge_and_unload()
m.save_pretrained("./cyber-witten-7b-merged"); AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct").save_pretrained("./cyber-witten-7b-merged")
EOF

git clone --depth 1 https://github.com/ggerganov/llama.cpp && pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py ./cyber-witten-7b-merged --outfile cyber-witten-7b-f16.gguf
cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build -j --target llama-quantize
llama.cpp/build/bin/llama-quantize cyber-witten-7b-f16.gguf cyber-witten-7b-Q4_K_M.gguf Q4_K_M
# download cyber-witten-7b-Q4_K_M.gguf (~4.7 GB) to the laptop, then kill the pod
```

## 5. Serve locally with ollama

```bash
cat > Modelfile <<'EOF'
FROM ./cyber-witten-7b-Q4_K_M.gguf
PARAMETER num_ctx 8192
EOF
ollama create cyber-witten-7b -f Modelfile
```

(`num_ctx 8192`: the serving prompt is ~5-6k tokens; ollama's 4096 default would
silently truncate the passages.)

## 6. Measure (the whole point)

```bash
.venv/bin/python -m evals.run_eval --providers ollama --model cyber-witten-7b            # distilled, naked
.venv/bin/python -m evals.run_eval --providers ollama --model cyber-witten-7b --guardrail
# compare against the base-model rows already in evals/results/full_run_merged.jsonl:
# qwen2.5:7b naked: correctness 0.42/2, cite-recall 0.00 | +guard: 0.53/2, cite-recall 0.64
```

Success looks like: naked cite-recall well above 0.00 (the distilled model
cites without being forced), and correctness above 0.53 (it anchors on the
right paper more often). The gold set stayed held out from training, so the
comparison is clean.
