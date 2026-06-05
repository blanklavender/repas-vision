import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf
from sam3.model.sam3_image_processor import Sam3Processor

# python .\sam3_test.py ..\canopy_detection\captures\capture_20260602_121019\color_20260602_121019.png

TEXT_PROMPT = "green plants"
CONFIDENCE_THRESHOLD = 0.5
CHECKPOINT_VERSION = "sam3"
DEVICE = "cuda"
MIN_AREA_PX = 0
LARGEST_ONLY = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Sam3PlantSegmenter:
    def __init__(self, prompt=TEXT_PROMPT, confidence=CONFIDENCE_THRESHOLD,
                 version=CHECKPOINT_VERSION, device=DEVICE):
        self.prompt = prompt
        self.device = device
        ckpt = download_ckpt_from_hf(version) if version == "sam3.1" else None
        self.model = build_sam3_image_model(device=device, checkpoint_path=ckpt)
        self.processor = Sam3Processor(self.model, device=device,
                                       confidence_threshold=confidence)

    def segment(self, image, prompt=None):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        prompt = self.prompt if prompt is None else prompt
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            self.processor.reset_all_prompts(state)
            state = self.processor.set_text_prompt(prompt=prompt, state=state)
        h, w = state["original_height"], state["original_width"]
        masks = state.get("masks")
        if masks is None or masks.shape[0] == 0:
            return np.zeros((h, w), dtype=bool)
        masks = masks.squeeze(1).to(torch.bool).cpu().numpy()
        if MIN_AREA_PX > 0:
            masks = masks[masks.reshape(masks.shape[0], -1).sum(1) >= MIN_AREA_PX]
        if masks.shape[0] == 0:
            return np.zeros((h, w), dtype=bool)
        if LARGEST_ONLY:
            return masks[masks.reshape(masks.shape[0], -1).sum(1).argmax()]
        return masks.any(axis=0)


if __name__ == "__main__":
    import sys
    seg = Sam3PlantSegmenter()
    img = Image.open(sys.argv[1]).convert("RGB")
    mask = seg.segment(img)
    print(mask.shape, mask.dtype, int(mask.sum()))