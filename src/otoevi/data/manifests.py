"""Read report manifests and check patient, report and image separation."""

import json
import re
from pathlib import Path


def patient_id(entry):
    if "patient_id" in entry and entry["patient_id"] is None:
        return None
    if entry.get("patient_id") is not None:
        return str(entry["patient_id"])
    report = Path(entry["filenames"][0]).stem.split("__v", 1)[0]
    match = re.fullmatch(r"oto_([^_]+)_[^_]+_(?:left|right)", report)
    if not match:
        raise ValueError("Supply a de-identified patient_id for each report")
    return match.group(1)


def load_manifest(path):
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must be a nonempty JSON list")
    ids = set()
    for entry in entries:
        if not entry.get("report_id") or entry["report_id"] in ids:
            raise ValueError("report_id must be nonempty and unique within a split")
        ids.add(entry["report_id"])
        if not isinstance(entry.get("caption"), str) or not entry["caption"].strip():
            raise ValueError("Training/evaluation reports need a nonempty caption")
        if not isinstance(entry.get("filenames"), list) or not entry["filenames"]:
            raise ValueError("Each report needs at least one real image")
        patient_id(entry)
    return entries


def audit_split(train, val):
    for field in ("report_id", "patient_id"):

        def values(entries):
            return {
                patient_id(e) if field == "patient_id" else e[field] for e in entries
            }

        if (values(train) - {None}) & (values(val) - {None}):
            raise ValueError("Train/validation overlap in " + field)

    def images(entries):
        return {name for e in entries for name in e["filenames"]}

    if images(train) & images(val):
        raise ValueError("Train/validation image filename overlap")
