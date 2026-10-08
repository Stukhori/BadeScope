"""Regenerate the landing preview using the app's frozen local models."""
from pathlib import Path
import json

from PIL import Image

from windblade_demo.constants import CLASS_LABELS, CHECKPOINT_STATE_FINGERPRINT
from windblade_demo.crops import contextual_crop
from windblade_demo.detector import load_proposal_detector, propose_regions
from windblade_demo.explain import generate_gradcam
from windblade_demo.inference import load_frozen_model, infer
from windblade_demo.visualization import annotate_proposals


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "app/assets/landing"
    output.mkdir(parents=True, exist_ok=True)
    detector = load_proposal_detector(root)
    model = load_frozen_model(root)
    cases = []
    for source_id in (1, 31, 71, 126, 234, 311, 434, 501):
        source = root / f"data/raw/wtbd/WT blade defect dataset/JPEGImages/{source_id}.jpg"
        destination = output if source_id == 1 else output / f"case_{source_id}"
        if build_case(root, source, destination, detector, model):
            cases.append({"id": source_id, "directory": "." if source_id == 1 else destination.name})
        if len(cases) == 4:
            break
    if len(cases) < 4:
        raise RuntimeError("At least four examples with detected regions are required.")
    (output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")


def build_case(root, source, output, detector, model):
    image = Image.open(source).convert("RGB")
    proposals = propose_regions(detector, image)
    if not proposals:
        return False
    output.mkdir(parents=True, exist_ok=True)
    proposal = proposals[0]
    crop = contextual_crop(image, proposal.box).model_input
    result = infer(model, crop)
    image.save(output / "source.jpg")
    annotate_proposals(image, proposals).save(output / "regions.jpg")
    crop.save(output / "crop.png")
    generate_gradcam(model, crop, result.predicted_class_id).overlay.save(output / "attention.png")
    (output / "prediction.json").write_text(json.dumps({
        "source": str(source.relative_to(root)),
        "checkpoint_state_fingerprint": CHECKPOINT_STATE_FINGERPRINT,
        "proposal_id": proposal.proposal_id,
        "box": list(proposal.box.as_tuple()),
        "predicted_label": result.predicted_label,
        "scores": dict(zip(CLASS_LABELS, result.scores)),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Built {source.name}: {result.predicted_label}", flush=True)
    return True


if __name__ == "__main__":
    main()
