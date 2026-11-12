"""Read a result directory: an optional manifest plus per-seed JSON records.

Records are passed through a MethodAdapter before the gates see them, so a
non-DeLP method can use its own column names without touching the checker.
"""
import json
import pathlib

from . import adapters

EXCLUDED = {"manifest.json", "report.json", "report.md", "ci-summary.md"}


def load_experiment(folder, adapter=None):
    """Return (manifest, records, notes).

    Accepted per-seed file shapes: a single record dict, a list of records, or a
    dict with a ``records`` list. Errored records are kept here (the gates decide
    what to do with them) but unreadable files are reported as notes.

    ``adapter`` may be an adapter instance, a registered name, or a mapping spec.
    When omitted, the manifest's ``adapter`` field decides; the DeLP ladder
    remains the default for backwards compatibility.
    """
    p = pathlib.Path(folder)
    if not p.is_dir():
        raise FileNotFoundError("not a directory: %s" % p)

    notes = []
    manifest_path = p / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            notes.append("manifest.json unreadable: %s" % exc)
    else:
        notes.append("no manifest.json: declaration-level gates will be skipped")

    try:
        spec = adapter if adapter is not None else manifest.get("adapter")
        adap = (spec if isinstance(spec, adapters.MethodAdapter)
                else adapters.resolve(spec))
    except (KeyError, TypeError) as exc:
        notes.append("adapter resolution failed (%s); using identity" % exc)
        adap = adapters.REGISTRY["identity"]
    if adap.name not in ("identity", "delp_ladder"):
        notes.append("adapter: %s (%s)" % (adap.name, adap.description))

    records = []
    raw = []
    for f in sorted(p.glob("*.json")):
        if f.name in EXCLUDED:
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            notes.append("%s unreadable: %s" % (f.name, exc))
            continue
        if isinstance(d, list):
            raw.extend(x for x in d if isinstance(x, dict))
        elif isinstance(d, dict) and isinstance(d.get("records"), list):
            raw.extend(x for x in d["records"] if isinstance(x, dict))
        elif isinstance(d, dict):
            raw.append(d)
    records = adap.normalize_records(raw)
    return manifest, records, notes
