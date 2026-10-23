#!/usr/bin/env bash
# ==============================================================================
#   SFT CoT 81 MILITARY CLASSES - ONE-SHOT PIPELINE FOR LINUX (CUDA)
#   Step 1: Check/install PyTorch >= 2.6 (CUDA) + Transformers/PEFT packages
#   Step 2: Pull Parquet dataset from HuggingFace Hub (skipped if already local)
#   Step 3: Train SFT with enable_thinking=True (ThinkingDataCollator)
#
#   Usage:
#     bash run_sft_qwen9b.sh                                  # 1 GPU, tu do VRAM
#     MODEL_ID=Qwen/Qwen3.5-4B bash run_sft_qwen9b.sh         # doi model
#     NUM_GPUS=2 bash run_sft_qwen9b.sh                       # DDP qua torchrun
#     TORCH_SOURCE=PYPI bash run_sft_qwen9b.sh                # torch tu PyPI mac dinh
#     EPOCHS=2 LR=1e-4 bash run_sft_qwen9b.sh                 # override hyper-param
# ==============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(pwd)"

# ----------------------------------------------------------------------------
# 1. CAU HINH THAM SO (moi cai co the override bang bien moi truong)
# ----------------------------------------------------------------------------
HF_DATASET_REPO="${HF_DATASET_REPO:-KhanhChien/military-81classes-sft-cot}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-${SCRIPT_DIR}/data_hf/military-81classes-sft-cot}"

# GPU >= 22 GB (RTX 3090/4090/A100/L40S): chay 9B 16-bit duoc.
# GPU 8-16 GB (RTX 4060/T4/L4/P100): nen MODEL_ID=Qwen/Qwen3.5-4B, script tu bat --use_4bit.
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/$(basename "${MODEL_ID}")-SFT-CoT-81Classes-thinking}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
SAVE_STEPS="${SAVE_STEPS:-200}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1280}"
NUM_GPUS="${NUM_GPUS:-1}"
# TORCH_SOURCE: CUDA_12_6 (mac dinh, tai cu the cu126) | PYPI (wheel mac dinh cua PyPI) | CPU (khong CUDA)
TORCH_SOURCE="${TORCH_SOURCE:-CUDA_12_6}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTHONUNBUFFERED=1

if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

# pip co the bi chan boi PEP 668 (Debian 12 / Ubuntu 24.04) -> thu --break-system-packages
pip_install() {
    if ! "${PY}" -m pip install -q "$@" 2>/dev/null; then
        "${PY}" -m pip install -q --break-system-packages "$@" 2>/dev/null \
            || "${PY}" -m pip install -q --user "$@"
    fi
}

echo "=============================================================================="
echo "  SFT CoT 81 Military Classes (Linux)"
echo "  MODEL_ID    = ${MODEL_ID}"
echo "  DATA        = ${LOCAL_DATA_DIR}"
echo "  OUTPUT_DIR  = ${OUTPUT_DIR}"
echo "  NUM_GPUS    = ${NUM_GPUS}"
echo "=============================================================================="

echo
echo "[1/3] Checking PyTorch >= 2.6 + CUDA and required packages..."
if ! "${PY}" -c "import sys, torch; v=tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if (v >= (2, 6) and torch.cuda.is_available()) else 1)" 2>/dev/null; then
    echo "  -> Installing PyTorch >= 2.6 (source: ${TORCH_SOURCE})..."
    case "${TORCH_SOURCE}" in
        CUDA_12_6) pip_install --upgrade "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu126 ;;
        CUDA_12_4) pip_install --upgrade "torch>=2.6" torchvision --index-url https://download.pytorch.org/whl/cu124 ;;
        PYPI)      pip_install --upgrade torch torchvision ;;
        *)         echo "[ERROR] TORCH_SOURCE khong hop le: ${TORCH_SOURCE} (dung CUDA_12_6 | CUDA_12_4 | PYPI)"; exit 1 ;;
    esac
fi
pip_install -U "transformers>=4.57" accelerate peft bitsandbytes \
    qwen-vl-utils datasets pyarrow huggingface_hub pillow tqdm
"${PY}" -c "import torch, transformers, peft; print('  -> Env OK: torch=' + torch.__version__ + ' | cuda=' + str(torch.cuda.is_available()) + ' | transformers=' + transformers.__version__ + ' | peft=' + peft.__version__)"

echo
echo "[2/3] Checking Parquet dataset (${HF_DATASET_REPO})..."
export HF_DATASET_REPO LOCAL_DATA_DIR
"${PY}" - <<'PYCHECK'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset

repo = os.environ["HF_DATASET_REPO"]
out = Path(os.environ["LOCAL_DATA_DIR"])
out.mkdir(parents=True, exist_ok=True)
pdir = out / "data"


def scan():
    if not pdir.is_dir():
        return [], []
    return (sorted(str(p) for p in pdir.glob("train-*.parquet")),
            sorted(str(p) for p in pdir.glob("validation-*.parquet")))


tf, vf = scan()
if len(tf) < 4 or len(vf) < 1:
    snapshot_download(repo_id=repo, repo_type="dataset",
                      allow_patterns=["data/*.parquet", "*.jsonl"],
                      local_dir=str(out))
    tf, vf = scan()
if not tf:
    raise SystemExit(f"Khong tim thay file .parquet nao trong {pdir}")

data_files = {"train": tf}
if vf:
    data_files["validation"] = vf
ds = load_dataset("parquet", data_files=data_files)
val_n = len(ds["validation"]) if "validation" in ds else 0
print(f"  -> Dataset OK: Train = {len(ds['train'])} | Val = {val_n} | sample image = {ds['train'][0]['image'].size}")
PYCHECK

# ----------------------------------------------------------------------------
# 3. Tu do VRAM -> bat QLoRA 4-bit neu < 22 GB
# ----------------------------------------------------------------------------
EXTRA_ARGS=()
read -r VRAM_GB GPU_NAME <<EOF2
$("${PY}" -c "import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'{p.total_memory / 1024**3:.1f}', p.name.replace(' ', '_'))
else:
    print('0.0', 'no_cuda')")
EOF2
echo "  -> GPU: ${GPU_NAME} | VRAM: ${VRAM_GB} GB"
if awk "BEGIN { exit !(${VRAM_GB} < 22) }"; then
    echo "  -> VRAM < 22 GB: auto-enabling --use_4bit (QLoRA NF4)"
    EXTRA_ARGS+=(--use_4bit)
else
    echo "  -> VRAM >= 22 GB: full 16-bit LoRA (BF16), no quantization"
fi

# Bao som neu model + 4-bit van vuot qua VRAM (tranh mat hang gio tai xong crash)
"${PY}" - "${MODEL_ID}" "${VRAM_GB}" "${EXTRA_ARGS[@]:-}" <<'PYPREFLIGHT'
import sys
from huggingface_hub import HfApi

model_id, vram_gb = sys.argv[1], float(sys.argv[2])
use_4bit = "--use_4bit" in sys.argv[3:]
try:
    info = HfApi().model_info(model_id, files_metadata=True)
    gb = sum(f.size or 0 for f in info.siblings
             if str(f.rfilename).endswith(".safetensors")) / 1024**3
except Exception as exc:
    print(f"  -> [preflight] Bo qua kiem tra kich thuoc ({type(exc).__name__})")
    sys.exit(0)
need = (gb * 0.5 + 2.5) if use_4bit else (gb + 3.0)
print(f"  -> [preflight] {model_id}: {gb:.1f} GB weights | uoc tinh can ~{need:.1f} GB tren {vram_gb:.1f} GB VRAM")
if need > vram_gb:
    print(f"[ERROR] GPU KHONG du cho model nay ({need:.1f} GB > {vram_gb:.1f} GB).")
    print("        Chay local: MODEL_ID=Qwen/Qwen3.5-4B bash run_sft_qwen9b.sh")
    print("        Hoac dung thong qua Kaggle Note GPU T4/P100 (16 GB) / L4 (24 GB).")
    sys.exit(1)
PYPREFLIGHT

echo
echo "[3/3] Starting SFT Training (enable_thinking=True) on ${MODEL_ID}..."
TRAIN_ARGS=(
    --model_id "${MODEL_ID}"
    --parquet_dir "${LOCAL_DATA_DIR}"
    --output_dir "${OUTPUT_DIR}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --grad_accum "${GRAD_ACCUM}"
    --lr "${LR}"
    --lora_r "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --save_steps "${SAVE_STEPS}"
    --max_seq_length "${MAX_SEQ_LEN}"
)
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node="${NUM_GPUS}" "${SCRIPT_DIR}/train_sft_qwen4b_81classes.py" "${TRAIN_ARGS[@]}"
else
    "${PY}" "${SCRIPT_DIR}/train_sft_qwen4b_81classes.py" "${TRAIN_ARGS[@]}"
fi

echo
echo "=============================================================================="
echo "  COMPLETED. Final LoRA adapter saved at:"
echo "  ${OUTPUT_DIR}/final_adapter"
echo "=============================================================================="
