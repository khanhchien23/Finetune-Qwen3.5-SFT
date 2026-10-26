"""Qwen3.5-9B SFT LoRA on RealTest crops with the exact SFT prompt.

Attach the RealTest dataset and checkpoint-668 as Kaggle Inputs. The prompt is embedded from the 9B training
Parquet, so no Parquet dataset is needed on Kaggle for inference.
With Kaggle T4 x2, the 16-bit model is sharded across both GPUs; with one T4
it offloads remaining layers to CPU RAM (much slower).
"""

import csv
import re
import sys
import time
from pathlib import Path

# Match the 9B training script's Kaggle torchao/PEFT compatibility workaround.
sys.modules["torchao"] = None
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
import peft.tuners.lora.torchao
peft.tuners.lora.torchao.is_torchao_available = lambda: False
from peft import PeftModel


# realtest_items() reads REALTEST_ROOT / "crops" and the manifest beside it.
REALTEST_ROOT = Path("/kaggle/input/datasets/chizus2602/anh-that/RealTest_Full_81Classes")
ADAPTER = Path(
    "/kaggle/input/models/khanhchien/model-sft-thinking/pytorch/default/1/"
    "Qwen3.5-9B-SFT-CoT-81Classes-thinking/checkpoint-668"
)
MODEL_ID = "Qwen/Qwen3.5-9B"
OUTPUT = Path("/kaggle/working/realtest_lora_predictions.csv")
LOG_OUTPUT = OUTPUT.with_suffix(".log")  # Human-readable thinking and final label per crop.
LIMIT = 5                # First 5 crops for a smoke test; 0 = all crops.
MAX_NEW_TOKENS = 512
LOAD_IN_4BIT = False     # True: faster if FP16 offloads; use separate OUTPUT (predictions can differ).
ATTENTION_IMPLEMENTATION = None  # Try "sdpa" only after benchmarking on your Kaggle T4s.
START_INDEX = 0          # 0-based start in sorted crops; split sessions into chunks.
STOP_INDEX = 0           # 0 = through LIMIT/all; otherwise exclusive 0-based stop.
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
GPU_HEADROOM_GIB = 3  # Leave room for image embeddings, attention and generation cache.
CPU_RAM_GIB = 24       # Set below available system RAM; disk is a last resort.


# Exact system_prompt and first user_prompt from the train shard used by
# run_sft_qwen9b.sh (KhanhChien/military-81classes-sft-cot). The train data
# contains four user_prompt variants; use this same variant for all RealTest
# crops and for both baseline/LoRA so evaluation remains comparable.
VALID_81_CLASSES = (
    "Air.A10", "Air.AH64", "Air.AS365", "Air.B2", "Air.B29", "Air.C130", "Air.C17",
    "Air.Casa212", "Air.CH47", "Air.CH53", "Air.F22", "Air.F35", "Air.IL76", "Air.J7",
    "Air.JH7", "Air.Ka27", "Air.Ka52", "Air.KC135", "Air.L39", "Air.Mi24", "Air.Mi28",
    "Air.Mi8", "Air.Mig29", "Air.Rafale", "Air.Su22", "Air.Su27", "Air.T6Texan",
    "Air.Yak130", "Air.Yak52", "Air.Z8", "Grd.2S1", "Grd.BM21", "Grd.BM27",
    "Grd.BM30", "Grd.BMP2", "Grd.Bradley", "Grd.BRDM2", "Grd.BTR152", "Grd.BTR60",
    "Grd.Gaz66", "Grd.Himars", "Grd.Humvee", "Grd.Kamaz", "Grd.Kraz255", "Grd.LAV25",
    "Grd.M113", "Grd.M1Abrams", "Grd.Maz537", "Grd.Merkava", "Grd.MTLB", "Grd.PT76",
    "Grd.SU100", "Grd.SU76", "Grd.T34", "Grd.T55", "Grd.T90", "Grd.Ural4320",
    "Grd.Zil131", "Grd.ZSU234", "Grd.ZSU57", "Sea.ArleighBurke", "Sea.Buyan",
    "Sea.GeraldFord", "Sea.Gorshkov", "Sea.Independence", "Sea.Kilo", "Sea.Kirov",
    "Sea.Molniya", "Sea.Nanuchka", "Sea.OsaII", "Sea.Ticonderoga",
    "Sea.Type002.Shandong", "Sea.Type022.Houbei", "Sea.Type051B.Luhai",
    "Sea.Type053H3.JiangweiII", "Sea.Type054A.JiangkaiII", "Sea.Type055.Renhai",
    "Sea.TypeAsahi", "Sea.TypeChengKung", "Sea.TypeKangDing", "Sea.Wasp",
)
SYSTEM_PROMPT = f"""You are an expert military analyst specializing in Fine-Grained Military Vehicle Recognition.
Your task is to inspect the provided image and classify the military vehicle into EXACTLY ONE of the following 81 target classes:

{", ".join(VALID_81_CLASSES)}

Follow this structured reasoning process before making your final decision:
1. Viewpoint & Environment: Identify the camera viewing angle and operational setting.
2. Coarse Category: Determine the primary domain (Air / Grd / Sea) and general vehicle type.
3. Fine-Grained Visual Features: Describe the distinctive physical characteristics strictly visible in the image.
4. Disambiguation & Conclusion: Rule out visually similar candidate types based on observed features and select the exact target class name.

Required output format:
<think>
Write your step-by-step visual analysis here...
</think>
<answer>Exact_Class_Name</answer>"""
USER_PROMPT = (
    "Inspect this military vehicle carefully. Provide a structured 4-step visual analysis "
    "inside <think>...</think> and conclude with the exact class label inside <answer>...</answer>."
)


def realtest_items(root):
    crop_dir = root / "crops"
    if not crop_dir.is_dir():
        raise FileNotFoundError(f"Missing RealTest crops directory: {crop_dir}")
    paths = sorted(p for p in crop_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f"No crop images found in {crop_dir}")
    manifest = root / "clean_81classes_full_manifest.csv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            labels = {}
            for record in csv.DictReader(stream):
                name = record["crop_file"]
                if name in labels:
                    raise ValueError(f"Duplicate crop_file in manifest: {name}")
                labels[name] = record["class_folder_name"]
        if set(labels) != {p.name for p in paths} or any(not v for v in labels.values()):
            raise ValueError("Manifest labels do not match crops; check REALTEST_ROOT")
    else:
        print(f"No {manifest.name}; writing predictions without ground truth/accuracy", flush=True)
        labels = {}
    return paths, labels


def parse_answer(raw):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"<answer>\s*(.*)", raw, re.DOTALL)
    return match.group(1).strip() if match else ""


def load_model_with_offload():
    """Preserve 16-bit weights; shard across visible GPUs before CPU offload."""
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No CUDA GPU visible to PyTorch")
    max_memory = {}
    for gpu in range(gpu_count):
        gpu_gib = torch.cuda.get_device_properties(gpu).total_memory / 1024**3
        gpu_budget = int(gpu_gib - GPU_HEADROOM_GIB)
        if gpu_budget < 4:
            raise RuntimeError(f"GPU {gpu} has only {gpu_gib:.1f} GiB; insufficient headroom")
        max_memory[gpu] = f"{gpu_budget}GiB"
        print(f"GPU {gpu}: {torch.cuda.get_device_name(gpu)} | "
              f"{gpu_gib:.1f} GiB total | {gpu_budget} GiB model budget", flush=True)
    max_memory["cpu"] = f"{CPU_RAM_GIB}GiB"
    offload_folder = Path("/kaggle/working/qwen9b_lora_offload")
    offload_folder.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading {MODEL_ID} ({dtype}, 4-bit={LOAD_IN_4BIT}) on {gpu_count} visible GPU(s) | "
          f"max_memory={max_memory}", flush=True)
    quantization = {}
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig
        quantization["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    try:
        attention = ({"attn_implementation": ATTENTION_IMPLEMENTATION}
                     if ATTENTION_IMPLEMENTATION else {})
        base = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=dtype, device_map="auto", max_memory=max_memory,
            offload_folder=str(offload_folder), offload_state_dict=True,
            low_cpu_mem_usage=True, trust_remote_code=True, **quantization, **attention,
        )
        model = PeftModel.from_pretrained(base, str(ADAPTER), is_trainable=False).eval()
    except (torch.OutOfMemoryError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            "Could not load 9B LoRA with 16-bit CPU offload. Restart Kaggle session "
            "after a failed load; check available CPU RAM/disk or use a >=24 GiB GPU."
        ) from exc
    used = set(base.hf_device_map.values())
    print(f"Loaded; device_map: {base.hf_device_map}", flush=True)
    if gpu_count >= 2 and 1 not in used and "cuda:1" not in used:
        print("WARNING: GPU 1 received no layers; check CUDA_VISIBLE_DEVICES/device_map", flush=True)
    if "cpu" in used or "disk" in used:
        print("WARNING: Some weights are offloaded to CPU/disk; inference will be slow", flush=True)
    return model


def load_processor():
    source = str(ADAPTER) if (ADAPTER / "processor_config.json").is_file() else MODEL_ID
    try:
        return AutoProcessor.from_pretrained(
            source, size={"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
            trust_remote_code=True,
        )
    except (TypeError, ValueError):
        return AutoProcessor.from_pretrained(
            source, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, trust_remote_code=True,
        )


def infer_one(path, processor, model):
    started = time.monotonic()
    with Image.open(path) as file_image:
        image = file_image.convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    if not prompt.endswith("<|im_start|>assistant\n<think>\n"):
        raise ValueError("Chat template does not match the SFT thinking prompt")
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to("cuda:0")
    generation_started = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    generation_seconds = time.monotonic() - generation_started
    generated_tokens = generated.shape[1] - inputs["input_ids"].shape[1]
    raw = "<think>\n" + processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=False,
    )[0]
    prediction = parse_answer(raw)
    del inputs, generated
    return prediction, raw, generated_tokens, generation_seconds, time.monotonic() - started


def record_result(path, result, labels, writer, stream, log_stream, index, total, gpu=None):
    prediction, raw, generated_tokens, generation_seconds, elapsed = result
    gt = labels.get(path.name, "")
    is_correct = prediction.strip().casefold() == gt.strip().casefold() if gt else ""
    writer.writerow({"file_name": path.name, "ground_truth": gt,
                     "prediction": prediction, "correct": is_correct, "raw_output": raw})
    stream.flush()
    log_stream.write(
        f"{'=' * 80}\n"
        f"File: {path.name}\n"
        f"Ground truth: {gt or '(unavailable)'}\n"
        f"Prediction: {prediction or '(no <answer> found)'}\n"
        f"Correct: {is_correct if gt else '(unavailable)'}\n"
        f"Tokens: {generated_tokens}/{MAX_NEW_TOKENS}\n"
        f"Raw model output (including <think> and <answer>):\n{raw}\n\n"
    )
    log_stream.flush()
    prefix = f"GPU {gpu} " if gpu is not None else ""
    print(f"[{index}/{total}] {prefix}Done {path.name} in {elapsed:.1f}s "
          f"(generate={generation_seconds:.1f}s, tokens={generated_tokens}/{MAX_NEW_TOKENS}): "
          f"prediction={prediction!r} gt={gt!r}", flush=True)
    if generated_tokens >= MAX_NEW_TOKENS and not prediction:
        print(f"WARNING: {path.name} reached MAX_NEW_TOKENS without <answer>; "
              "increase the limit or inspect raw_output", flush=True)
    return bool(is_correct)


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("Enable GPU in Kaggle Notebook Settings; 9B BF16/FP16 requires enough VRAM")
    if LIMIT < 0 or MAX_NEW_TOKENS < 1 or START_INDEX < 0 or STOP_INDEX < 0:
        raise ValueError("LIMIT/START_INDEX/STOP_INDEX must be >= 0 and MAX_NEW_TOKENS >= 1")
    if ATTENTION_IMPLEMENTATION not in (None, "sdpa"):
        raise ValueError("ATTENTION_IMPLEMENTATION must be None or 'sdpa'")
    if LOAD_IN_4BIT and OUTPUT.name == "realtest_lora_predictions.csv":
        raise ValueError("Set OUTPUT to a separate CSV for 4-bit results before running")
    if not (ADAPTER / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Checkpoint missing adapter_config.json: {ADAPTER}; check Kaggle Input layout")
    paths, labels = realtest_items(REALTEST_ROOT)
    total = min(len(paths), LIMIT) if LIMIT else len(paths)
    end = min(total, STOP_INDEX) if STOP_INDEX else total
    if START_INDEX >= end:
        raise ValueError(f"Empty range: START_INDEX={START_INDEX}, end={end}; check LIMIT/STOP_INDEX")
    selected = paths[START_INDEX:end]
    print(f"RealTest: indices [{START_INDEX}:{end}] ({len(selected)}/{len(paths)} crops) | "
          "loading processor and model...", flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file_name", "ground_truth", "prediction", "correct", "raw_output"]
    completed = set()
    if OUTPUT.is_file() and OUTPUT.stat().st_size:
        with OUTPUT.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != fieldnames:
                raise ValueError(f"Unexpected CSV columns in {OUTPUT}; refusing to overwrite existing results")
            for row in reader:
                if row["file_name"]:
                    completed.add(row["file_name"])
    selected_names = {path.name for path in selected}
    correct = 0
    if labels and completed:
        with OUTPUT.open(newline="", encoding="utf-8") as stream:
            correct = sum(row["correct"] == "True" and row["file_name"] in selected_names
                          for row in csv.DictReader(stream))
    remaining = [path for path in selected if path.name not in completed]
    done_in_range = len(selected) - len(remaining)
    print(f"Resume: {done_in_range} crops saved; {len(remaining)} crops remaining in selected range", flush=True)
    if not remaining:
        print(f"Saved {OUTPUT} (nothing left to infer); log: {LOG_OUTPUT}")
        return
    processor = load_processor()
    model = load_model_with_offload()
    print(f"Starting LoRA inference: {len(selected)} crops -> {OUTPUT}; log -> {LOG_OUTPUT}", flush=True)
    with OUTPUT.open("a", newline="", encoding="utf-8") as stream, LOG_OUTPUT.open(
            "a", encoding="utf-8") as log_stream:
        writer = csv.DictWriter(stream, fieldnames=["file_name", "ground_truth", "prediction", "correct", "raw_output"])
        if stream.tell() == 0:
            writer.writeheader()
            stream.flush()
        for index, path in enumerate(tqdm(remaining, desc="LoRA inference", unit="img"), 1):
            print(f"[{index}/{len(remaining)}] Starting {path.name} "
                  f"(max_new_tokens={MAX_NEW_TOKENS})...", flush=True)
            result = infer_one(path, processor, model)
            correct += record_result(path, result, labels, writer, stream, log_stream, index, len(remaining))
    print(f"Saved {OUTPUT} and {LOG_OUTPUT}")
    if labels and START_INDEX == 0 and end == len(paths):
        print(f"Accuracy on {len(paths)} crops: {correct}/{len(paths)} = {correct / len(paths):.2%}")


if __name__ == "__main__":
    run()