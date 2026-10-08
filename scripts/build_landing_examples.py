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
    source = root / "data/raw/wtbd/WT blade defect dataset/JPEGImages/1.jpg"
    image = Image.open(source).convert("RGB")
    proposals = propose_regions(load_proposal_detector(root), image)
    if not proposals:
        raise RuntimeError("The example image must contain a detector proposal.")
    proposal = proposals[0]
    crop = contextual_crop(image, proposal.box).model_input
    model = load_frozen_model(root)
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


if __name__ == "__main__":
    main()
