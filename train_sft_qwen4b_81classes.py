#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Script huấn luyện SFT (Supervised Fine-Tuning) với Chain-of-Thought (CoT)
cho mô hình Qwen3.5-4B / Qwen3-VL-4B-Instruct / Qwen2.5-VL-3B-Instruct
trên bộ dữ liệu 81 lớp phương tiện quân sự (`sft_cot_81classes_all.jsonl`).

Đặc điểm tối ưu hóa cho Kaggle (2x T4 16GB hoặc P100 16GB) & Local GPU:
1. Tự động dò tìm file `sft_cot_81classes_all.jsonl` (hoặc `train.jsonl`/`val.jsonl`) và thư mục chứa ảnh gốc.
2. Tự động cắt vùng Bounding Box (`bbox`: [x, y, w, h]) từ ảnh gốc có đệm ngữ cảnh nhẹ (10% padding)
   để khớp 100% với phân phối ảnh crop của tập test `RealTest_Full_81Classes`.
3. Che do TRAIN: enable_thinking=True (ThinkingDataCollator):
   - Prompt ket thuc tai assistant + mo the think; loss chi tinh tren reasoning,
     dong think, answer va ket thuc luot (mask -100 toan bo system/user/anh).
4. Hỗ trợ chạy đa GPU tốc độ cao với `torchrun --nproc_per_node=2` (DDP) + LoRA 16-bit (`fp16`/`bf16`)
   hoặc 1 GPU, hoàn thành 11,868 mẫu trong ~1.8 - 2.5 giờ/epoch trên Kaggle 2x T4.
"""

import os
import sys
import re
import json
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Bật expandable_segments ngay từ đầu để chống phân mảnh bộ nhớ CUDA trên T4 15GB
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Ngăn lỗi xung đột phiên bản torchao==0.10.0 cài sẵn trên Kaggle với peft mới
sys.modules["torchao"] = None

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
import peft.tuners.lora.torchao
peft.tuners.lora.torchao.is_torchao_available = lambda: False
from peft import LoraConfig, get_peft_model, TaskType

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_UTILS = True
except ImportError:
    HAS_QWEN_UTILS = False


# ==============================================================================
# 1. CẤU HÌNH MẶC ĐỊNH
# ==============================================================================
DEFAULT_MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3.5-4B")
DEFAULT_OUTPUT_DIR = "/kaggle/working/Qwen4B-SFT-CoT-81Classes" if os.path.exists("/kaggle/working") else "./Qwen4B-SFT-CoT-81Classes"

# Kích thước ảnh thực tế trong bộ mô phỏng (640x640 -> sau khi crop BBox diện tích từ 64,800 đến 139,500 px):
# Đặt MIN_PIXELS = 64 * 28 * 28 (~50,176 px) và MAX_PIXELS = 256 * 28 * 28 (~200,704 px)
# giúp giữ nguyên 100% độ phân giải gốc của vùng crop mà không bị upscale thừa hay tràn VRAM.
# Thu muc chua file nay (cross-platform, thay cho duong dan Windows hardcode)
SCRIPT_DIR = Path(__file__).resolve().parent

MIN_PIXELS = 64 * 28 * 28    # 50,176 pixels
MAX_PIXELS = 256 * 28 * 28   # 200,704 pixels


# ==============================================================================
# 2. TỰ ĐỘNG DÒ TÌM FILE JSONL & THƯ MỤC ẢNH TRÊN KAGGLE / LOCAL
# ==============================================================================
def find_dataset_files(custom_jsonl: Optional[str] = None, custom_img_root: Optional[str] = None):
    """Tự động tìm file sft_cot_81classes_all.jsonl và thư mục ảnh trên Kaggle hoặc Local."""
    jsonl_candidates = []
    if custom_jsonl:
        jsonl_candidates.append(Path(custom_jsonl))
    jsonl_candidates.extend([
        Path("/kaggle/input/datasets/khnhchin/cot-jsonl/sft_cot_81classes_all.jsonl"),
        Path("/kaggle/input/datasets/khnhchin/cot-jsonl/sft_cot_81classes_train.jsonl"),
        Path("/kaggle/input/cot-jsonl/sft_cot_81classes_all.jsonl"),
        Path("./sft_cot_81classes_all.jsonl"),
        SCRIPT_DIR / "sft_cot_81classes_all.jsonl",
        SCRIPT_DIR / "data_hf" / "military-81classes-sft-cot" / "sft_cot_81classes_all.jsonl",
        Path("/kaggle/working/sft_cot_81classes_all.jsonl"),
    ])

    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            for fname in files:
                if fname in ("sft_cot_81classes_all.jsonl", "sft_cot_81classes_train.jsonl"):
                    jsonl_candidates.append(Path(root) / fname)

    jsonl_path = next((p for p in jsonl_candidates if p.is_file()), None)
    if jsonl_path is None:
        raise FileNotFoundError(
            "Không tìm thấy `sft_cot_81classes_all.jsonl`! "
            "Hãy gắn Dataset chứa file JSONL vào Kaggle hoặc truyền `--jsonl_path`."
        )

    # Tìm thư mục chứa các file ảnh (vd: sim_step1280.camera_0.png)
    img_dir_candidates = []
    if custom_img_root:
        img_dir_candidates.append(Path(custom_img_root))
    img_dir_candidates.extend([
        Path("/kaggle/input/datasets/khnhchin/anh-mo-phong-2/images/images"),
        Path("/kaggle/input/datasets/khnhchin/anh-mo-phong-2/images"),
        Path("/kaggle/input/datasets/khnhchin/anh-mo-phong-2"),
        Path("/kaggle/input/anh-mo-phong-2/images/images"),
        Path("/kaggle/input/anh-mo-phong-2/images"),
        Path("/kaggle/input/anh-mo-phong-2"),
        jsonl_path.parent / "images" / "images",
        jsonl_path.parent / "images",
        jsonl_path.parent,
        Path("/kaggle/working/images"),
        SCRIPT_DIR / "images" / "images",
        SCRIPT_DIR / "images",
        SCRIPT_DIR,
    ])

    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if any(f.endswith(".camera_0.png") for f in files[:20]):
                img_dir_candidates.append(Path(root))

    # Xây dựng bảng ánh xạ tên file ảnh -> đường dẫn thực tế để tra cứu O(1)
    image_lookup = {}
    img_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    for cand_dir in img_dir_candidates:
        if cand_dir.is_dir():
            try:
                with os.scandir(cand_dir) as it:
                    for entry in it:
                        if entry.is_file() and os.path.splitext(entry.name)[1].lower() in img_exts:
                            if entry.name not in image_lookup:
                                image_lookup[entry.name] = entry.path
            except Exception:
                pass

    return jsonl_path, image_lookup


def load_and_split_jsonl(jsonl_path: Path, image_lookup: Dict[str, str], val_ratio: float = 0.05):
    """
    Đọc toàn bộ bản ghi từ sft_cot_81classes_all.jsonl, đối chiếu file ảnh tồn tại,
    và chia Stratified Train / Val theo từng lớp.
    """
    records = []
    missing_cnt = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            fname = item.get("file_name") or Path(item.get("image", "")).name
            resolved_path = image_lookup.get(fname)

            if not resolved_path:
                cand = jsonl_path.parent / item.get("image", "")
                if cand.is_file():
                    resolved_path = str(cand)

            if resolved_path and os.path.exists(resolved_path):
                item["resolved_image_path"] = resolved_path
                records.append(item)
            else:
                missing_cnt += 1

    if not records:
        raise RuntimeError(
            f"Đã đọc {jsonl_path} nhưng không tìm thấy file ảnh nào khớp! "
            f"(Số ảnh thiếu: {missing_cnt}). Hãy kiểm tra thư mục ảnh trên Kaggle."
        )

    # Chia Stratified Train / Val theo gt_class
    by_class = defaultdict(list)
    for r in records:
        by_class[r["gt_class"]].append(r)

    rng = random.Random(42)
    train_data, val_data = [], []
    for cls_name, cls_items in sorted(by_class.items()):
        items_copy = list(cls_items)
        rng.shuffle(items_copy)
        if val_ratio > 0 and len(items_copy) > 2:
            n_val = max(1, int(round(len(items_copy) * val_ratio)))
            val_data.extend(items_copy[:n_val])
            train_data.extend(items_copy[n_val:])
        else:
            train_data.extend(items_copy)

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    return train_data, val_data, missing_cnt


def load_parquet_dataset(parquet_dir_or_repo: str):
    """
    Đọc trực tiếp từ thư mục chứa các file `.parquet` (trên Kaggle / Vast.ai / Local)
    hoặc tên repo HuggingFace Hub. Cột `image` tự động trả về `PIL.Image.Image`!
    """
    from datasets import load_dataset
    p = Path(parquet_dir_or_repo)
    if p.is_dir():
        parquet_files = sorted(p.rglob("*.parquet"))
        train_files = [str(f) for f in parquet_files if "train" in f.name.lower()]
        val_files = [str(f) for f in parquet_files if "val" in f.name.lower()]
        if not train_files and parquet_files:
            train_files = [str(f) for f in parquet_files]
        data_files = {"train": train_files}
        if val_files:
            data_files["validation"] = val_files
        ds = load_dataset("parquet", data_files=data_files)
    else:
        ds = load_dataset(parquet_dir_or_repo)

    train_ds = ds["train"]
    val_ds = ds["validation"] if "validation" in ds else None
    return train_ds, val_ds


# ==============================================================================
# 3. DATASET CẮT BBOX ĐỘNG (DYNAMIC CROP) & DATA COLLATOR MASK LOSS (-100)
# ==============================================================================
class MilitaryCoTSFTDataset(Dataset):
    """
    Dataset phục vụ huấn luyện SFT (hỗ trợ cả 2 nguồn):
    1. Đọc từ file Parquet (`datasets.Dataset` với cột `image` đã là `PIL.Image` cắt sẵn BBox).
    2. Đọc từ JSONL + thư mục ảnh gốc (`resolved_image_path` và cắt theo `bbox` động).
    """

    def __init__(self, data: Any, crop_bbox: bool = True, pad_ratio: float = 0.08):
        self.data = data
        self.crop_bbox = crop_bbox
        self.pad_ratio = pad_ratio

    def __len__(self) -> int:
        return len(self.data)

    def _load_and_crop_image(self, item: Dict[str, Any]) -> Image.Image:
        # Nếu đọc từ Parquet (cột 'image' đã là đối tượng PIL.Image hoặc dict bytes)
        raw_img = item.get("image")
        if isinstance(raw_img, Image.Image):
            return raw_img.convert("RGB")
        if isinstance(raw_img, dict) and raw_img.get("bytes"):
            import io
            return Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")

        img = Image.open(item["resolved_image_path"]).convert("RGB")
        bbox = item.get("bbox")
        if self.crop_bbox and bbox and len(bbox) == 4:
            x, y, w, h = [float(v) for v in bbox]
            pad_x = w * self.pad_ratio
            pad_y = h * self.pad_ratio
            x1 = max(0, int(math.floor(x - pad_x)))
            y1 = max(0, int(math.floor(y - pad_y)))
            x2 = min(img.width, int(math.ceil(x + w + pad_x)))
            y2 = min(img.height, int(math.ceil(y + h + pad_y)))
            if x2 > x1 + 4 and y2 > y1 + 4:
                img = img.crop((x1, y1, x2, y2))
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        img = self._load_and_crop_image(item)

        if "messages" in item:
            raw_msgs = item["messages"]
            sys_text = raw_msgs[0]["content"]
            user_content = raw_msgs[1]["content"]
            if isinstance(user_content, list):
                user_text = next((c["text"] for c in user_content if c.get("type") == "text"), "")
            else:
                user_text = str(user_content)
            assistant_text = raw_msgs[2]["content"]
        else:
            # Định dạng cột trực tiếp từ Parquet (system_prompt, user_prompt, assistant_cot)
            sys_text = item["system_prompt"]
            user_text = item["user_prompt"]
            assistant_text = item["assistant_cot"]

        prompt_messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_text}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        ]

        return {
            "prompt_messages": prompt_messages,
            "full_messages": full_messages,
            "image": img,
        }


@dataclass
class ThinkingDataCollator:
    processor: Any
    max_seq_length: int = 1280

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_texts, prompt_texts = [], []
        for feature in features:
            assistant = feature["full_messages"][-1]["content"][0]["text"]
            if not assistant.startswith("<think>\n") or "</think>" not in assistant or "<answer>" not in assistant:
                raise ValueError("Assistant target phải bắt đầu bằng <think>\\n và có </think>, <answer>.")

            # Template chuyển thẻ <think> trong target sang reasoning_content,
            # nên chỉ có một thẻ <think> (không chèn thinking rỗng).
            full = self.processor.apply_chat_template(
                feature["full_messages"], tokenize=False,
                add_generation_prompt=False, enable_thinking=True,
            )
            prompt = self.processor.apply_chat_template(
                feature["prompt_messages"], tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
            if not prompt.endswith("<|im_start|>assistant\n<think>\n"):
                raise ValueError(f"Prompt chưa bật thinking: {prompt[-100:]!r}")
            if not full.startswith(prompt):
                raise ValueError(
                    "Full text và prompt lệch nhau trước phần reasoning; "
                    "không thể mask loss an toàn. Kiểm tra chat template/model."
                )
            full_texts.append(full)
            prompt_texts.append(prompt)

        if HAS_QWEN_UTILS:
            images, videos = process_vision_info([f["full_messages"] for f in features])
            image_kwargs = {"images": images, "videos": videos}
        else:
            image_kwargs = {"images": [f["image"] for f in features]}

        batch = self.processor(
            text=full_texts, **image_kwargs, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts, **image_kwargs, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        for i in range(len(features)):
            prompt_len = int(prompt_batch["attention_mask"][i].sum().item())
            indices = (batch["attention_mask"][i] == 1).nonzero(as_tuple=True)[0]
            if not len(indices):
                raise ValueError("Mẫu train không có token đầu vào.")
            start = int(indices[0].item())
            # Xác nhận prefix token khớp, tránh mask nhầm image tokens hay
            # token reasoning khi tokenizer tự thêm special tokens.
            prompt_ids = prompt_batch["input_ids"][i][prompt_batch["attention_mask"][i] == 1]
            if not torch.equal(batch["input_ids"][i, start:start + prompt_len], prompt_ids):
                raise ValueError("Prompt token không là prefix của full input_ids; không thể mask an toàn.")
            labels[i, start:start + prompt_len] = -100
            if not (labels[i] != -100).any():
                raise ValueError(
                    "Response bị cắt hết do max_seq_length; hãy tăng max_seq_length "
                    "hoặc giảm kích thước ảnh."
                )
        batch["labels"] = labels
        return batch


# ==============================================================================
# 4. HÀM KHỞI TẠO MÔ HÌNH QWEN-4B + LORA (PEFT) + CHUNKED CROSS-ENTROPY LOSS
# ==============================================================================
class MemoryEfficientTrainer(Trainer):
    """
    Trainer tùy chỉnh tránh lỗi OOM 2.00 GiB ở hàm ForCausalLMLoss của Qwen3.5-4B:
    - Không truyền `labels` vào `model(**inputs)` để ngăn HuggingFace tự động nhân/ép kiểu
      toàn bộ tensor logits [B, L, 248320] sang fp32 (chiếm đúng 2.00 GiB VRAM!).
    - Chỉ trích xuất các vị trí token có `labels != -100` (tức phần <think>...</answer> ~195 tokens)
      rồi mới tính Cross-Entropy Loss -> chỉ tốn ~180 MB thay vì 2.00 GiB!
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        if labels is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        logits = outputs.logits  # [B, L, V] ở fp16/bf16
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        valid_mask = shift_labels.ne(-100)
        if valid_mask.any():
            active_logits = shift_logits[valid_mask]   # Chỉ ~ [195, 248320] thay vì [1600, 248320]
            active_labels = shift_labels[valid_mask]
            loss = torch.nn.functional.cross_entropy(
                active_logits.float(), active_labels, reduction="mean"
            )
        else:
            loss = shift_logits.sum() * 0.0

        return (loss, outputs) if return_outputs else loss


def build_model_and_processor(model_id: str, lora_r: int = 16, lora_alpha: int = 32, use_qlora_4bit: bool = False):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    gpu_name = torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else ""
    # bf16 yêu cầu card Ampere/Ada/Hopper trở lên (A100/A6000/L40/L40S/H100/4090/5090...);
    # T4/P100 (Volta/Pascal/Turing) không có bf16 unit -> giữ fp16.
    # Ampere tro len (SM >= 8.0) co bf16; dung capability thuc te thay vi doi ten
    # vi "RTX 4060 Laptop" khong khop bat ky ten nao trong list cu -> dung fp16.
    try:
        _cap = torch.cuda.get_device_capability(local_rank) if torch.cuda.is_available() else (0, 0)
        use_bf16 = _cap[0] >= 8
    except Exception:
        use_bf16 = any(k in gpu_name.upper() for k in [
            "A100", "A800", "A6000", "A5000", "A40", "H100", "H200",
            "L4", "L40", "3090", "4090", "5090", "TI",
        ])
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if local_rank == 0:
        print(f"--> Đang tải Processor & Model: {model_id}")
        print(f"--> GPU: {gpu_name} | Dtype: {compute_dtype} | QLoRA 4-bit: {use_qlora_4bit}")

    # Chuẩn mới của transformers v5+: truyền size={'shortest_edge': MIN_PIXELS, 'longest_edge': MAX_PIXELS}
    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
            trust_remote_code=True,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS,
            trust_remote_code=True,
        )
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "right"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": {"": local_rank} if torch.cuda.is_available() else None,
    }

    if use_qlora_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["dtype"] = compute_dtype

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    except TypeError:
        model_kwargs.pop("dtype", None)
        model_kwargs["torch_dtype"] = compute_dtype
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)

    # Bật Gradient Checkpointing để tiết kiệm 60% bộ nhớ Activation trên T4 16GB
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Cấu hình LoRA trên các tầng Linear của Language + Projector
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    if local_rank == 0:
        model.print_trainable_parameters()

    return model, processor, use_bf16


# ==============================================================================
# 5. CHƯƠNG TRÌNH HUẤN LUYỆN CHÍNH
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="SFT CoT Training for Qwen-4B/9B on 81 Military Classes")
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--parquet_dir", type=str, default=None, help="Đường dẫn thư mục chứa file .parquet hoặc repo HuggingFace")
    parser.add_argument("--jsonl_path", type=str, default=None)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--max_seq_length", type=int, default=1280)
    parser.add_argument("--crop_bbox", action="store_true", default=True)
    parser.add_argument("--no_crop_bbox", action="store_false", dest="crop_bbox")
    parser.add_argument("--use_4bit", action="store_true", default=False)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # 1. Tìm dữ liệu (Ưu tiên đọc từ Parquet nếu có --parquet_dir, hoặc tự động tìm JSONL + Ảnh)
    if args.parquet_dir:
        train_records, val_records = load_parquet_dataset(args.parquet_dir)
        data_source_str = f"Parquet ({args.parquet_dir})"
    else:
        jsonl_path, image_lookup = find_dataset_files(args.jsonl_path, args.image_root)
        train_records, val_records, missing_cnt = load_and_split_jsonl(
            jsonl_path, image_lookup, val_ratio=args.val_ratio
        )
        data_source_str = str(jsonl_path)

    if local_rank == 0:
        print("=" * 78)
        print("CẤU HÌNH DỮ LIỆU HUẤN LUYỆN SFT CoT (81 LỚP QUÂN SỰ)")
        print("=" * 78)
        print(f"- Nguồn dữ liệu           : {data_source_str}")
        print(f"- Số mẫu Train            : {len(train_records):,} mẫu")
        print(f"- Số mẫu Val              : {len(val_records) if val_records else 0:,} mẫu")
        print(f"- Cắt theo BBox (Dynamic) : {args.crop_bbox}")
        print(f"- Thư mục lưu Checkpoint  : {args.output_dir}")
        print("=" * 78)

    train_dataset = MilitaryCoTSFTDataset(train_records, crop_bbox=args.crop_bbox)
    val_dataset = MilitaryCoTSFTDataset(val_records, crop_bbox=args.crop_bbox) if val_records else None

    # 2. Khởi tạo Model + LoRA + Processor
    model, processor, use_bf16 = build_model_and_processor(
        args.model_id,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_qlora_4bit=args.use_4bit,
    )
    collator = ThinkingDataCollator(processor=processor, max_seq_length=args.max_seq_length)
    # Kiểm tra template/masking ngay trước khi khởi chạy vòng lặp train.
    collator([train_dataset[0]])

    # 3. Thiết lập TrainingArguments tối ưu cho Kaggle 2x T4 / P100 (tương thích transformers v5+)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.save_steps if val_dataset else None,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
    )

    trainer = MemoryEfficientTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # 4. Tự động Resume từ checkpoint gần nhất nếu có
    last_ckpt = None
    if os.path.isdir(args.output_dir):
        ckpts = [
            os.path.join(args.output_dir, d)
            for d in os.listdir(args.output_dir)
            if d.startswith("checkpoint-")
        ]
        if ckpts:
            ckpts.sort(key=lambda x: int(x.split("-")[-1]))
            last_ckpt = ckpts[-1]
            if local_rank == 0:
                print(f"--> Phát hiện Checkpoint cũ, tiếp tục huấn luyện từ: {last_ckpt}")

    trainer.train(resume_from_checkpoint=last_ckpt)

    # 5. Lưu LoRA Adapter và Processor cuối cùng
    if local_rank == 0:
        final_adapter_dir = os.path.join(args.output_dir, "final_adapter")
        os.makedirs(final_adapter_dir, exist_ok=True)
        trainer.model.save_pretrained(final_adapter_dir)
        processor.save_pretrained(final_adapter_dir)
        print(f"\n✅ Đã hoàn tất huấn luyện SFT! Trọng số LoRA Adapter được lưu tại: {final_adapter_dir}")


if __name__ == "__main__":
    main()
