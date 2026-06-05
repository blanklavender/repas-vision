import argparse
import os
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# python sam3_segment.py --image color_391252.png --prompt "green plant with white pot"

CHECKPOINT = "./sam3_weights/sam3.pt"
ALPHA = 0.45
CONTOUR_THICKNESS = 2
LABEL_SCALE = 0.6
MAX_DISPLAY_W = 1400
PALETTE = [
    (0, 255, 0), (255, 64, 64), (64, 128, 255), (255, 200, 0),
    (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 255, 128),
]


def load_model(confidence):
    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    bpe_path = os.path.join(sam3_root, "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=CHECKPOINT, load_from_HF=False)
    return Sam3Processor(model, confidence_threshold=confidence)


def segment(processor, image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        print(f"[SAM3] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        ctx = nullcontext()
        print("[SAM3] CPU")
    with ctx:
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)
    return state


def extract(state, shape):
    h, w = shape
    out = []
    for i in range(len(state["scores"])):
        mask = state["masks"][i].squeeze(0).cpu().numpy()
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(bool)
        score = state["scores"][i]
        if isinstance(score, torch.Tensor):
            score = score.item()
        out.append((mask, float(score)))
    out.sort(key=lambda m: -m[1])
    return out


def annotate(color_rgb, masks):
    overlay = color_rgb.copy()
    for i, (mask, score) in enumerate(masks):
        color = np.array(PALETTE[i % len(PALETTE)])
        overlay[mask] = (overlay[mask] * (1 - ALPHA) + color * ALPHA).astype(np.uint8)

    for i, (mask, score) in enumerate(masks):
        color = tuple(int(c) for c in PALETTE[i % len(PALETTE)])
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, CONTOUR_THICKNESS)
        ys, xs = np.where(mask)
        if not len(xs):
            continue
        label = f"#{i} {score:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SCALE, 2)
        x, y = int(xs.min()), max(int(ys.min()), th + base + 2)
        cv2.rectangle(overlay, (x, y - th - base - 2), (x + tw + 6, y), color, -1)
        cv2.putText(overlay, label, (x + 3, y - base), cv2.FONT_HERSHEY_SIMPLEX,
                    LABEL_SCALE, (0, 0, 0), 2, cv2.LINE_AA)
    return overlay


def show(image_bgr, title):
    h, w = image_bgr.shape[:2]
    if w > MAX_DISPLAY_W:
        s = MAX_DISPLAY_W / w
        image_bgr = cv2.resize(image_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, image_bgr)
    print("[VIEW] press any key in the window to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    color_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise SystemExit(f"cannot read image: {args.image}")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    processor = load_model(args.confidence)
    state = segment(processor, args.image, args.prompt)
    masks = extract(state, color_rgb.shape[:2])

    print(f"[SAM3] {len(masks)} object(s) for '{args.prompt}' (conf>={args.confidence})")
    for i, (mask, score) in enumerate(masks):
        print(f"  #{i}  score={score:.3f}  pixels={int(mask.sum()):,}")
    if not masks:
        raise SystemExit("no detections")

    result_bgr = cv2.cvtColor(annotate(color_rgb, masks), cv2.COLOR_RGB2BGR)
    out_path = args.out or f"{os.path.splitext(args.image)[0]}_seg.png"
    cv2.imwrite(out_path, result_bgr)
    print(f"[SAVE] {out_path}")

    if not args.no_show:
        show(result_bgr, args.prompt)


if __name__ == "__main__":
    main()