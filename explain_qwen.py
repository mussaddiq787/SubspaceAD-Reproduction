"""
explain_qwen.py - Grounded defect explanations for SubspaceAD outputs.

Idea: SubspaceAD DECIDES (image score vs threshold); Qwen3-VL only EXPLAINS.
Inputs : per_image_scores.csv + intro_overlays/ produced by a SubspaceAD run
         (run main.py with --save_intro_overlays on the qwen-explain branch)
Outputs: qwen_explanations.csv (one row per evaluated image) + printed summary

Usage:
  python explain_qwen.py --run_dir results/qwen_run --per_type 5
  python explain_qwen.py --run_dir results/qwen_run --per_type 1 --categories bottle   # quick test
  python explain_qwen.py --run_dir results/qwen_run --per_type 5 --ungrounded          # ablation
"""
import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import precision_recall_curve
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
IMG_SIZE = 448  # resize overlays before sending them to Qwen (fewer tokens, less VRAM)


# ---------------------------------------------------------------- helpers
def best_f1_threshold(labels, scores):
    """Threshold that maximises F1 on the given scores.
    NOTE: computed on the test set (MVTec has no validation set), so it is an
    'oracle' threshold. State this as a limitation in the report."""
    p, r, t = precision_recall_curve(labels, scores)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[int(f1[:-1].argmax())])


def norm(name):
    """'Broken Large' / 'broken-large' -> 'broken_large' so names can be compared."""
    return re.sub(r"[\s\-]+", "_", str(name).strip().lower())


def parse_json(text):
    """Pull the first {...} block out of the model answer; empty dict if it fails."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def build_prompt(category, defect_types, decision, score, thr, grounded, input_mode="overlay"):
    if input_mode == "crop":
        return build_crop_prompt(category, defect_types, decision, score, thr, grounded)
    types = ", ".join(defect_types)
    if grounded:
        facts = (
            f"An anomaly detector analysed this image. Its decision: {decision} "
            f"(anomaly score {score:.3f}, decision threshold {thr:.3f}).\n"
            "Coloured regions mark the most unusual patches. The colours are relative "
            "within this image, so they appear even on normal images; trust the decision.\n"
            "If the decision is NORMAL, answer defect_type \"none\".\n"
        )
    else:
        facts = "Inspect this image for manufacturing defects. If there is none, answer defect_type \"none\".\n"
    return (
        f"You are an industrial visual inspection assistant.\nProduct: {category}.\n"
        f"{facts}"
        f"Possible defect types: {types}.\n"
        "If there is a defect, choose the single most likely type from the list.\n"
        'Answer ONLY with JSON: {"defect_type": "...", "location": "...", '
        '"description": "one short sentence", "severity": "none|low|medium|high"}'
    )


def build_crop_prompt(category, defect_types, decision, score, thr, grounded):
    if grounded:
        facts = (
            f"An anomaly detector analysed this image. Its decision: {decision} "
            f"(anomaly score {score:.3f}, decision threshold {thr:.3f}).\n"
            "Trust the detector decision. If the decision is NORMAL, answer defect_type \"none\".\n"
        )
    else:
        facts = "Inspect this image for manufacturing defects. If there is none, answer defect_type \"none\".\n"
    return (
        f"You are an industrial visual inspection assistant.\nProduct: {category}.\n"
        f"{facts}"
        "Image 1 is the full product, image 2 is a zoomed view of the region the detector flagged. "
        "Describe the physical defect you see on the object; do not mention colours of any heatmap.\n"
        f"Possible defect types: {', '.join(defect_types)}.\n"
        "If there is a defect, choose the single most likely type from the list.\n"
        'Answer ONLY with JSON: {"defect_type": "...", "location": "...", '
        '"description": "one short sentence", "severity": "none|low|medium|high"}'
    )


def prepare_crop_images(original_path, overlay_path, full_size=IMG_SIZE, margin=0.15):
    """Recover the rendered region, then crop only untouched original pixels.

    Match the renderer's default PIL resize exactly before comparing PNG pixels.
    This locates the denoised displayed region, not a raw anomaly-map argmax.
    Bounding boxes are right/bottom exclusive. Margin is per side, clipped to
    the image. Empty masks use the full image and are recorded explicitly.
    """
    with Image.open(original_path) as opened:
        original = opened.convert("RGB")
    with Image.open(overlay_path) as opened:
        overlay = opened.convert("RGB")
    resized = original.resize(overlay.size)
    changed = np.any(np.asarray(overlay) != np.asarray(resized), axis=2)
    mask_box = Image.fromarray(changed.astype(np.uint8) * 255).getbbox()
    width, height = overlay.size
    fallback = "empty_overlay_difference" if mask_box is None else ""
    if mask_box is None:
        mask_box = (0, 0, width, height)
    left, top, right, bottom = mask_box
    pad_x = math.ceil((right - left) * margin)
    pad_y = math.ceil((bottom - top) * margin)
    padded = (max(0, left - pad_x), max(0, top - pad_y),
              min(width, right + pad_x), min(height, bottom + pad_y))
    ow, oh = original.size
    box = (math.floor(padded[0] * ow / width), math.floor(padded[1] * oh / height),
           min(ow, math.ceil(padded[2] * ow / width)),
           min(oh, math.ceil(padded[3] * oh / height)))
    full = original.resize((full_size, full_size))
    crop = original.crop(box).resize((IMG_SIZE, IMG_SIZE))
    details = {
        "input_mode": "crop", "full_image_size": full_size, "crop_image_size": IMG_SIZE,
        "crop_margin_fraction_per_side": margin,
        "hotspot_bbox_overlay": json.dumps(mask_box),
        "crop_bbox_overlay": json.dumps(padded), "crop_bbox_original": json.dumps(box),
        "hotspot_changed_pixels": int(changed.sum()), "crop_fallback": fallback,
        "crop_area_fraction": (box[2] - box[0]) * (box[3] - box[1]) / (ow * oh),
    }
    return [full, crop], details


# ---------------------------------------------------------------- model
def load_model():
    # fp16, not bf16: Turing GPUs (sm_75) have no native bf16 support
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


@torch.no_grad()
def ask(model, processor, image, prompt):
    images = list(image) if isinstance(image, (list, tuple)) else [image]
    messages = [{"role": "user", "content": [
        *[{"type": "image", "image": item} for item in images],
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=150, do_sample=False)  # greedy = reproducible
    new_tokens = out[:, inputs["input_ids"].shape[1]:]                   # drop the prompt
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="outdir of the SubspaceAD run")
    ap.add_argument("--per_type", type=int, default=5, help="images per defect type (incl. good)")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--ungrounded", action="store_true", help="do not give Qwen the detector decision")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input_mode", choices=["overlay", "crop"], default="overlay",
                    help="overlay keeps v1 unchanged; crop sends original full image and hotspot crop")
    ap.add_argument("--crop_full_size", type=int, choices=[448, 336], default=448,
                    help="crop mode only: reduce full view to 336 after CUDA OOM; crop stays 448")
    args = ap.parse_args()

    csv_path = next(Path(args.run_dir).rglob("per_image_scores.csv"))
    run_dir = csv_path.parent
    df = pd.read_csv(csv_path)
    if args.input_mode == "crop" and args.categories is None:
        reference = run_dir / "qwen_explanations_grounded.csv"
        if not reference.is_file():
            raise ValueError("Crop mode needs --categories or saved v1 grounded results to reuse its cohort")
        args.categories = list(pd.read_csv(reference).category.unique())
        print(f"Crop mode reuses v1 categories: {args.categories}")
    if args.categories:
        df = df[df.category.isin(args.categories)]
    print(f"Loaded {len(df)} image scores from {csv_path}")

    # per-category threshold and list of defect types (ground-truth folder names)
    thr = {c: best_f1_threshold(g.label, g.img_score) for c, g in df.groupby("category")}
    types = {c: sorted(t for t in g.defect_type.unique() if t != "good")
             for c, g in df.groupby("category")}

    # sample N images per (category, defect type) with a fixed seed
    rng = random.Random(args.seed)
    picks = []
    for (c, t), g in df.groupby(["category", "defect_type"]):
        rows = g.to_dict("records")
        rng.shuffle(rows)
        picks += rows[: args.per_type]
    print(f"Evaluating {len(picks)} images | grounded={not args.ungrounded}")

    if args.input_mode == "crop" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    model, processor = load_model()
    results = []
    for i, r in enumerate(picks, 1):
        c, p = r["category"], Path(r["path"])
        overlay = run_dir / "intro_overlays" / c / f"{p.parent.name}_{p.name}"
        if not overlay.exists():
            print(f"  missing overlay: {overlay}")
            continue
        crop_details = {}
        if args.input_mode == "crop":
            image, crop_details = prepare_crop_images(p, overlay, args.crop_full_size)
        else:
            image = Image.open(overlay).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        decision = "DEFECTIVE" if r["img_score"] >= thr[c] else "NORMAL"
        prompt = build_prompt(c, types[c], decision, r["img_score"], thr[c], not args.ungrounded,
                              input_mode=args.input_mode)
        raw = ask(model, processor, image, prompt)
        ans = parse_json(raw)
        pred = norm(ans.get("defect_type", "unparsed"))
        true = "none" if r["defect_type"] == "good" else norm(r["defect_type"])
        results.append({
            **r, **crop_details, "threshold": thr[c], "decision": decision,
            "pred_type": pred, "correct_type": pred == true,
            "location": ans.get("location", ""), "description": ans.get("description", ""),
            "severity": ans.get("severity", ""), "raw": raw,
        })
        print(f"[{i}/{len(picks)}] {c}/{r['defect_type']}: decision={decision} -> Qwen: {pred}")

    out = pd.DataFrame(results)
    tag = "ungrounded" if args.ungrounded else "grounded"
    if args.input_mode == "crop":
        tag = f"crop_{tag}"
    out_path = run_dir / f"qwen_explanations_{tag}.csv"
    out.to_csv(out_path, index=False)
    if args.input_mode == "crop" and torch.cuda.is_available():
        torch.cuda.synchronize()
        memory = {
            "input_mode": "crop", "full_image_size": args.crop_full_size,
            "crop_image_size": IMG_SIZE, "dtype": str(model.dtype),
            "gpu_name": torch.cuda.get_device_name(),
            "device_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "note": "PyTorch peaks include model loading and generation, not all driver/display memory",
        }
        out_path.with_suffix(".memory.json").write_text(json.dumps(memory, indent=2) + "\n")
        print(f"GPU peak allocated: {memory['peak_allocated_bytes'] / 2**30:.3f} GiB; "
              f"reserved: {memory['peak_reserved_bytes'] / 2**30:.3f} GiB")

    # ---------------- summary
    defective = out[out.label == 1]
    good = out[out.label == 0]
    print("\n=== Summary ===")
    print(f"Detector correct on sampled images: {((out.decision == 'DEFECTIVE') == (out.label == 1)).mean():.1%}")
    print(f"Defect-type accuracy (defective images): {defective.correct_type.mean():.1%}")
    for c, g in defective.groupby("category"):
        chance = 1 / max(len(types[c]), 1)
        print(f"  {c:12s} {g.correct_type.mean():.1%}  (chance {chance:.0%}, n={len(g)})")
    print(f"Good images described as defect-free: {(good.pred_type == 'none').mean():.1%} (n={len(good)})")
    print(f"Unparseable answers: {(out.pred_type == 'unparsed').mean():.1%}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()