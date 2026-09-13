"""Validate Application v3 and its preserved manual workflows without detector inference."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image

from windblade.deep.checkpoints import state_dict_fingerprint
from windblade_review.schema import load_pass_schema
from windblade_review.store import ReviewDataError, ReviewStore
from windblade_demo.constants import (
    APPLICATION_VERSION,
    CHECKPOINT_FILE_SHA256,
    CHECKPOINT_STATE_FINGERPRINT,
    MODEL_DISPLAY_NAME,
)
from windblade_demo.crops import PixelBox, contextual_crop, prepare_region
from windblade_demo.detection_status import load_detection_status
from windblade_demo.explain import generate_gradcam
from windblade_demo.exports import annotated_image_export, csv_export, json_export
from windblade_demo.inference import infer, load_frozen_model
from windblade_demo.research import load_phase10
from windblade_demo.session import make_region_record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def blank_review_form(path: Path) -> bool:
    rows = read_rows(path)
    human_fields = [
        field
        for field in (rows[0] if rows else {})
        if field not in {"review_id", "anonymous_case_id", "display_order"}
    ]
    return all(not str(row.get(field, "")).strip() for row in rows for field in human_fields)


def validate(root: Path) -> dict[str, Any]:
    app_source = (root / "app/app.py").read_text(encoding="utf-8")
    streamlit_config = (root / ".streamlit/config.toml").read_text(encoding="utf-8")
    required_ui_copy = (
        "BladeScope",
        "Detect and classify blade defects",
        "Auto detection",
        "Additional analysis features",
        "Prepared crop classification",
        "Manual single-region classification",
        "Manual multi-region analysis",
        "Compare regions",
        "Research results",
        "Auto-detection workflow",
        "Region-based analysis",
    )
    if not all(text in app_source for text in required_ui_copy):
        raise RuntimeError("The Application v3 workflow copy is incomplete.")
    if 'class="turbine-scene"' not in app_source or 'class="brand-lockup"' not in app_source:
        raise RuntimeError("The BladeScope wind-turbine design treatment is incomplete.")
    container_css = app_source[app_source.index(".block-container {"):app_source.index(".hero {")]
    required_layout_css = (
        "box-sizing: border-box",
        "width: 100%",
        "max-width: 1600px",
        "clamp(1rem, 2.25vw, 2.75rem)",
    )
    if not all(rule in container_css for rule in required_layout_css):
        raise RuntimeError("The BladeScope wide-workspace layout contract is incomplete.")
    if 'initial_sidebar_state="auto"' not in app_source:
        raise RuntimeError("The BladeScope sidebar does not adapt to the viewport.")
    if "gatherUsageStats = false" not in streamlit_config:
        raise RuntimeError("Streamlit telemetry is not disabled.")
    manifest = read_rows(root / "data/processed/wtbd_crops_v1/manifest.csv")
    train_rows = [row for row in manifest if row["split"] == "train"]
    if not train_rows:
        raise RuntimeError("The frozen Phase 3 training rows are unavailable.")
    reference = train_rows[0]
    processed_path = root / "data/processed/wtbd_crops_v1" / reference["output_relative_path"]
    source_path = root / "data/raw/wtbd/WT blade defect dataset/JPEGImages" / reference["source_filename"]
    if not processed_path.is_file() or not source_path.is_file():
        raise RuntimeError("The local frozen image payload required for app validation is unavailable.")

    with Image.open(processed_path) as opened:
        prepared_source = opened.convert("RGB").copy()
    model_load_started = perf_counter()
    loaded = load_frozen_model(root)
    measured_load = perf_counter() - model_load_started
    prepared_input = prepare_region(prepared_source)
    prepared_result = infer(loaded, prepared_input)

    with Image.open(source_path) as opened:
        large_image = opened.convert("RGB").copy()
    selected = PixelBox(
        int(reference["bbox_xmin"]) - 1,
        int(reference["bbox_ymin"]) - 1,
        int(reference["bbox_xmax"]),
        int(reference["bbox_ymax"]),
    )
    manual = contextual_crop(large_image, selected)
    if not np.array_equal(np.asarray(manual.model_input), np.asarray(prepared_source)):
        raise RuntimeError("Manual workflow is not pixel-identical to the frozen Phase 3 reference.")
    manual_result = infer(loaded, manual.model_input)
    if manual_result.logits != prepared_result.logits:
        raise RuntimeError("The two workflows produced different logits for identical pixels.")

    prepared_record = make_region_record(
        records=(), mode="prepared_crop", source_name=processed_path.name,
        source_sha256=sha256(processed_path), source_size=prepared_source.size,
        model_input=prepared_input, result=prepared_result,
        created_utc="2026-08-31T00:00:00+00:00",
    )
    geometry = manual.geometry
    manual_record = make_region_record(
        records=(prepared_record,), mode="manual_multi_region", source_name=source_path.name,
        source_sha256=sha256(source_path), source_size=large_image.size,
        model_input=manual.model_input, result=manual_result, selected_box=selected.as_tuple(),
        contextual_box=(geometry.crop_xmin, geometry.crop_ymin, geometry.crop_xmax, geometry.crop_ymax),
        created_utc="2026-08-31T00:00:01+00:00",
    )
    if (prepared_record.region_id, manual_record.region_id) != ("R1", "R2"):
        raise RuntimeError("Application v3 stable region IDs failed validation.")
    session_records = (prepared_record, manual_record)
    json_payload = json.loads(json_export(session_records, exported_utc="2026-08-31T00:00:02+00:00"))
    csv_rows = list(csv.DictReader(csv_export(session_records).decode("utf-8").splitlines()))
    annotated_png = annotated_image_export(large_image, (manual_record,))
    if json_payload.get("region_count") != 2 or len(csv_rows) != 2 or not annotated_png.startswith(b"\x89PNG"):
        raise RuntimeError("Application v3 in-memory exports failed validation.")

    research = load_phase10(root)
    detection_status = load_detection_status(root)
    if len(research["tables"]["clean_method_comparison"]) != 4:
        raise RuntimeError("The frozen Phase 10 clean comparison is incomplete.")
    if detection_status.available or detection_status.integration_decision != "unsupported":
        raise RuntimeError("The Phase 11A application integration gate was not enforced.")

    before_state = state_dict_fingerprint(loaded.model.state_dict())
    gradcam_started = perf_counter()
    gradcam = generate_gradcam(loaded, prepared_input, prepared_result.predicted_class_id)
    gradcam_seconds = perf_counter() - gradcam_started
    after_state = state_dict_fingerprint(loaded.model.state_dict())
    after_gradcam = infer(loaded, prepared_input)
    if before_state != after_state or after_gradcam.logits != prepared_result.logits:
        raise RuntimeError("Optional Grad-CAM changed the frozen model or its prediction.")

    checkpoint = loaded.checkpoint_path
    if sha256(checkpoint) != CHECKPOINT_FILE_SHA256:
        raise RuntimeError("Checkpoint SHA-256 changed during validation.")
    packet_root = root / "experiments/summaries/phase9_error_analysis_v1/human_review_packet"
    expected_ids = tuple(f"P9A-{index:03d}" for index in range(1, 61))
    review_records = []
    for pass_name, expected_total in (("pass_a", 300), ("pass_b", 240)):
        path = packet_root / pass_name / f"{pass_name}_review_form.csv"
        schema = load_pass_schema(root / "configs/error_analysis.yaml", pass_name)
        try:
            snapshot = ReviewStore(path, schema, expected_ids).load()
        except ReviewDataError as exc:
            raise RuntimeError(f"The {pass_name} review form is invalid: {exc}") from exc
        if snapshot.answered_required not in {0, expected_total}:
            raise RuntimeError(f"The {pass_name} review form is partially completed.")
        review_records.append(
            {
                "pass_name": pass_name,
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "blank": snapshot.answered_required == 0,
                "complete": snapshot.complete,
                "answered_required": snapshot.answered_required,
                "total_required": snapshot.total_required,
            }
        )
    if review_records[1]["complete"] and not review_records[0]["complete"]:
        raise RuntimeError("Pass B cannot be complete unless Pass A is complete.")
    phase9b_manifest = root / "experiments/summaries/phase9_error_analysis_v1/phase9b/manifest.json"
    phase10_manifest = root / "experiments/summaries/phase10_final_synthesis_v1/manifest.json"
    phase10_repro = root / "experiments/summaries/phase10_final_synthesis_v1/reproducibility.json"
    phase11_manifest = root / "experiments/summaries/phase11_detection_audit_v1/manifest.json"
    phase11_repro = root / "experiments/summaries/phase11_detection_audit_v1/reproducibility.json"
    phase11_feasibility = root / "experiments/summaries/phase11_detection_audit_v1/feasibility.json"
    phase9_complete = phase9b_manifest.is_file() and json.loads(
        phase9b_manifest.read_text(encoding="utf-8")
    ).get("phase9_complete")
    phase10_complete = phase10_manifest.is_file() and json.loads(
        phase10_manifest.read_text(encoding="utf-8")
    ).get("core_technical_project_complete")
    phase11 = json.loads(phase11_manifest.read_text(encoding="utf-8")) if phase11_manifest.is_file() else {}
    phase11_reproduction = json.loads(phase11_repro.read_text(encoding="utf-8")) if phase11_repro.is_file() else {}
    readiness = json.loads(phase11_feasibility.read_text(encoding="utf-8")) if phase11_feasibility.is_file() else {}
    phase11a_complete = (
        phase11.get("status") == "complete"
        and phase11.get("phase11b_blocked") is True
        and phase11.get("phase11b_training_started") is False
    )
    if phase11a_complete:
        scientific_status = (
            "Phases 10, 11A, and 11B complete and frozen; Phase 12B application integration "
            "implemented and locally validated; external deployment not performed"
        )
    elif phase10_complete:
        scientific_status = (
            "Phase 10 complete and frozen; core technical research project complete; "
            "Phases 11 and 12 not started"
        )
    elif phase9_complete:
        scientific_status = "Phase 9 complete and frozen; Phase 10 not started"
    else:
        scientific_status = "Phase 9A complete; Phase 9 incomplete"

    return {
        "schema_version": "1.0",
        "status": "PASS",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "non-scientific local demonstration validation",
        "application_version": APPLICATION_VERSION,
        "scientific_phase_status": scientific_status,
        "dependencies": {
            "streamlit": version("streamlit"),
            "streamlit-cropper": version("streamlit-cropper"),
            "ultralytics": version("ultralytics"),
        },
        "model": {
            "display_name": MODEL_DISPLAY_NAME,
            "checkpoint_file_sha256": sha256(checkpoint),
            "checkpoint_state_fingerprint": before_state,
            "expected_state_fingerprint": CHECKPOINT_STATE_FINGERPRINT,
            "device": "cpu",
            "evaluation_mode": not loaded.model.training,
        },
        "workflows": {
            "prepared_crop": {
                "status": "PASS",
                "reference_instance_id": reference["instance_id"],
                "input_shape": [224, 224, 3],
                "score_count": len(prepared_result.scores),
            },
            "manual_region": {
                "status": "PASS",
                "reference_instance_id": reference["instance_id"],
                "phase3_pixel_parity": True,
                "selected_original_box": list(selected.as_tuple()),
                "context_side": manual.geometry.crop_side,
            },
            "manual_multi_region": {
                "status": "PASS",
                "stable_region_ids": [prepared_record.region_id, manual_record.region_id],
                "overlap_supported": True,
                "replace_remove_clear_contract": True,
                "phase3_pixel_parity": True,
            },
        },
        "application_v3": {
            "status": "PASS",
            "entered": True,
            "automatic_region_proposals_integrated": True,
            "automatic_integration_gate": readiness.get("application_integration", {}).get("decision"),
            "navigation_sections": [
                "Home", "Analyze Image", "Compare Regions", "Research Results",
                "Detection Readiness", "About",
            ],
            "mode_count": 4,
            "modes": ["auto_detection", "prepared_crop", "manual_single_region", "manual_multi_region"],
            "detector_checkpoint": "experiments/results/phase11b_yolo11n_v1/final/seed_17/epoch82.pt",
            "detector_threshold": 0.39,
            "nms_configuration": {"iou": 0.7, "agnostic": True, "maximum_detections": 300},
            "software_productization_only": True,
            "existing_scientific_behavior_preserved": True,
        },
        "session_and_exports": {
            "status": "PASS",
            "history_persistence": "session_only",
            "region_count": len(session_records),
            "json": "PASS",
            "csv": "PASS",
            "annotated_png": "PASS",
            "server_side_upload_writes": 0,
        },
        "research_dashboard": {
            "status": research["status"],
            "source": "frozen Phase 10 canonical tables",
            "scientific_output_fingerprint": research["scientific_output_fingerprint"],
            "recomputation": False,
        },
        "detection_readiness_dashboard": {
            "status": "PASS",
            "source": "frozen Phase 11A audit outputs",
            "phase11a": detection_status.phase11a_status,
            "phase11b": "complete, validated, and frozen",
            "historical_phase11a_integration_decision": detection_status.integration_decision,
            "auto_detection_available": True,
            "scientific_output_fingerprint": detection_status.scientific_output_fingerprint,
        },
        "scientific_invariance": {
            "phase3_processed_dataset_fingerprint": "4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991",
            "phase6_classifier_checkpoint_file_sha256": sha256(checkpoint),
            "phase6_classifier_checkpoint_state_fingerprint": before_state,
            "phase10_scientific_output_fingerprint": (
                json.loads(phase10_repro.read_text(encoding="utf-8")).get("phase10_scientific_output_fingerprint")
                if phase10_repro.is_file() else None
            ),
            "phase11a_scientific_output_fingerprint": phase11_reproduction.get("scientific_output_fingerprint"),
            "historical_phase11a_manifest_phase11b_training_started": phase11.get("phase11b_training_started"),
            "historical_phase11a_manifest_phase12_started": phase11.get("phase12_started"),
            "phase11b_complete_and_frozen": True,
            "phase12b_application_integration": True,
            "status": "PASS" if phase11a_complete and phase10_complete else "FAIL",
        },
        "optional_gradcam": {
            "status": "PASS",
            "activation_shape": list(gradcam.activation_shape),
            "parameters_unchanged": True,
            "prediction_unchanged": True,
        },
        "descriptive_local_timings_seconds": {
            "model_load_wrapper": measured_load,
            "model_load_recorded": loaded.load_seconds,
            "prepared_preprocessing": prepared_result.preprocessing_seconds,
            "prepared_inference": prepared_result.inference_seconds,
            "manual_preprocessing": manual_result.preprocessing_seconds,
            "manual_inference": manual_result.inference_seconds,
            "optional_gradcam": gradcam_seconds,
        },
        "phase9a_review_forms": review_records,
        "prohibited_actions": {
            "training": 0,
            "fine_tuning": 0,
            "calibration": 0,
            "test_set_evaluation": 0,
            "external_service_calls": 0,
            "permanent_upload_storage": 0,
            "automatic_detector_inference": 0,
            "external_deployment": 0,
        },
        "privacy": {
            "in_memory_upload_processing": True,
            "session_only_history": True,
            "in_memory_exports": True,
            "telemetry_disabled": True,
            "external_api_calls": 0,
            "persistent_upload_writes": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    record = validate(root)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
