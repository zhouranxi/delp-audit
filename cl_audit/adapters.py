"""MethodAdapter: keep DeLP-specific field names out of the six gates.

The gates operate on a small internal schema (``arm``, ``ntask``, ``lam``,
``gain``, ``seed``, ``final_avg``, ``val_final_avg``, ``bwt``). Nothing in
``checks.py`` or ``grid.py`` should care whether the caller calls its method
column ``method``, its task count ``n_tasks`` or its penalty ``ewc_lambda``.

An adapter only renames fields and fills defaults. It never invents measurements:
missing metrics stay missing, so the corresponding gate reports SKIP rather
than a fabricated pass.
"""
from collections import OrderedDict

INTERNAL_FIELDS = ("arm", "ntask", "lam", "gain", "seed", "final_avg",
                   "val_final_avg", "bwt")


class MethodAdapter:
    """Base adapter. ``field_map`` maps external key -> internal key."""

    name = "identity"
    description = "records already use the cl-audit schema"

    def __init__(self, field_map=None, defaults=None, name=None):
        self.field_map = OrderedDict(field_map or {})
        self.defaults = dict(defaults or {})
        if name:
            self.name = name

    def normalize_record(self, record):
        if not isinstance(record, dict):
            return record
        out = dict(record)
        for src, dst in self.field_map.items():
            if src in out:
                out[dst] = out[src]
                if src != dst:
                    out.pop(src, None)
        for key, value in self.defaults.items():
            out.setdefault(key, value)
        return out

    def normalize_records(self, records):
        return [self.normalize_record(r) for r in records]

    def missing_fields(self, records):
        """Internal fields absent from every record (never an error by itself)."""
        present = set()
        for r in records:
            present.update(k for k in INTERNAL_FIELDS if k in r)
        return [k for k in INTERNAL_FIELDS if k not in present]

    def describe(self):
        return {"name": self.name, "description": self.description,
                "field_map": dict(self.field_map), "defaults": self.defaults}


class DelpLadderAdapter(MethodAdapter):
    name = "delp_ladder"
    description = ("DeLP mechanism ladder; accepts hgain as an alias for gain "
                   "and defaults the gain axis to 1.0 when it was not swept")

    def __init__(self):
        super().__init__(field_map={"hgain": "gain"},
                         defaults={"gain": 1.0})

    def normalize_record(self, record):
        out = super().normalize_record(record)
        # hgain is an implementation detail; keep it only if it adds information.
        if isinstance(out, dict) and "hgain" in out and "gain" in out:
            out.pop("hgain", None)
        return out


class GenericEwcAdapter(MethodAdapter):
    """Example non-DeLP adapter: a plain EWC / SGD result table.

    No dendritic segment, no context label, no MNIST assumption, no gain axis.
    """

    name = "ewc_generic"
    description = "generic regulariser study (plain EWC vs SGD), one penalty axis"

    def __init__(self):
        super().__init__(
            field_map={
                "method": "arm",
                "n_tasks": "ntask",
                "ewc_lambda": "lam",
                "penalty": "lam",
                "run_id": "seed",
                "avg_acc": "final_avg",
                "heldout_acc": "val_final_avg",
                "backward_transfer": "bwt",
            },
            defaults={"gain": 1.0})


REGISTRY = {
    "identity": MethodAdapter(),
    "delp_ladder": DelpLadderAdapter(),
    "ewc_generic": GenericEwcAdapter(),
}


def from_mapping(mapping, name="custom"):
    """Build an adapter from a plain {external: internal} dict."""
    return MethodAdapter(field_map=mapping, name=name,
                         description="user-supplied field mapping")


def get_adapter(spec=None):
    """Resolve a string, a mapping spec, or an adapter instance.

    A manifest may say ``"adapter": "ewc_generic"`` or
    ``"adapter": {"name": "custom", "field_map": {"method": "arm", ...}}``.
    """
    if spec is None:
        return REGISTRY["delp_ladder"]
    if isinstance(spec, MethodAdapter):
        return spec
    if isinstance(spec, str):
        if spec not in REGISTRY:
            raise KeyError("unknown adapter %r; available: %s"
                           % (spec, ", ".join(sorted(REGISTRY))))
        return REGISTRY[spec]
    if isinstance(spec, dict):
        name = spec.get("name", "custom")
        fm = spec.get("field_map") or {}
        defaults = spec.get("defaults") or {}
        return MethodAdapter(field_map=fm, defaults=defaults, name=name)
    raise TypeError("adapter must be a name, a mapping spec, or a MethodAdapter")


def resolve(manifest_spec=None, cli_spec=None):
    """CLI overrides the manifest; the DeLP ladder is the backward-compatible default."""
    return get_adapter(cli_spec if cli_spec is not None else manifest_spec)
