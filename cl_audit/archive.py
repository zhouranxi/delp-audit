"""Verify a shipped per-seed results archive without writing anything to disk.

A result set is only reproducible if (a) the tarball matches its published hash,
(b) every file listed in SHA256SUMS.txt is present in the tarball with exactly
that hash, and (c) there are no unlisted members hiding in the tarball. CI runs
this so a later edit cannot silently replace the evidence behind the README.
"""
import hashlib
import json
import pathlib
import tarfile


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _parse_sums(text):
    """Return {path: hash}; tolerate CRLF and the two-space sha256sum format."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "  " in line:
            digest, name = line.split("  ", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            digest, name = parts
        out[name.strip().lstrip("./")] = digest.strip().lower()
    return out


def verify_archive(folder, tar_name=None):
    """Verify <folder>/<tar>.tar.gz, its .sha256 and SHA256SUMS.txt.

    Returns a JSON-serialisable dict.  `ok` is False on any problem.
    """
    p = pathlib.Path(folder)
    res = {"folder": str(p), "ok": False, "problems": [], "tar": "",
           "tar_bytes": 0, "tar_sha256": "", "declared_tar_sha256": "",
           "n_sums": 0, "n_members": 0, "n_verified": 0,
           "missing": [], "extra": [], "mismatch": []}
    if not p.is_dir():
        res["problems"].append("not a directory: %s" % p)
        return res

    tars = sorted(p.glob("*.tar.gz")) if tar_name is None else [p / tar_name]
    if len(tars) != 1 or not tars[0].is_file():
        res["problems"].append("expected exactly one .tar.gz in %s, found %d"
                               % (p, len(tars)))
        return res
    tar = tars[0]
    res["tar"] = tar.name
    raw = tar.read_bytes()
    res["tar_bytes"] = len(raw)
    res["tar_sha256"] = _sha256_bytes(raw)

    side = pathlib.Path(str(tar) + ".sha256")
    if not side.is_file():
        res["problems"].append("missing %s" % side.name)
    else:
        declared = _parse_sums(side.read_text(encoding="utf-8"))
        want = declared.get(tar.name) or (next(iter(declared.values()))
                                          if declared else None)
        res["declared_tar_sha256"] = want or ""
        if not want or want.lower() != res["tar_sha256"]:
            res["problems"].append("tarball hash mismatch: declared %s, computed %s"
                                   % (want, res["tar_sha256"]))

    sums_path = p / "SHA256SUMS.txt"
    if not sums_path.is_file():
        res["problems"].append("missing SHA256SUMS.txt")
        return res
    sums = _parse_sums(sums_path.read_text(encoding="utf-8"))
    res["n_sums"] = len(sums)
    if not sums:
        res["problems"].append("SHA256SUMS.txt contains no entries")
        return res

    seen = set()
    try:
        with tarfile.open(str(tar), "r:gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name = member.name.lstrip("./")
                if not name.endswith(".json"):
                    continue
                res["n_members"] += 1
                seen.add(name)
                f = tf.extractfile(member)
                if f is None:
                    res["mismatch"].append(name)
                    continue
                got = _sha256_bytes(f.read())
                if name not in sums:
                    res["extra"].append(name)
                elif sums[name] != got:
                    res["mismatch"].append(name)
                else:
                    res["n_verified"] += 1
    except (tarfile.TarError, OSError) as exc:
        res["problems"].append("cannot read tarball: %s" % exc)
        return res

    res["missing"] = sorted(set(sums) - seen)
    if res["missing"]:
        res["problems"].append("%d listed files are absent from the tarball"
                               % len(res["missing"]))
    if res["extra"]:
        res["problems"].append("%d unlisted .json members in tarball"
                               % len(res["extra"]))
    if res["mismatch"]:
        res["problems"].append("%d files fail their SHA256" % len(res["mismatch"]))

    mpath = p / "manifest.json"
    if not mpath.is_file():
        res["problems"].append("missing manifest.json")
    else:
        try:
            m = json.loads(mpath.read_text(encoding="utf-8"))
            missing_keys = [k for k in ("effects", "boundary", "selfcheck")
                            if k not in m]
            if missing_keys:
                res["problems"].append("manifest.json lacks %s"
                                       % ", ".join(missing_keys))
        except (ValueError, OSError) as exc:
            res["problems"].append("manifest.json unreadable: %s" % exc)

    res["ok"] = not res["problems"]
    return res


def render_verify(res):
    lines = [
        "# cl-audit verify — %s" % res["folder"],
        "",
        "**%s** — %d SHA256SUMS entries, %d tarball JSON members, "
        "%d hashes verified." % ("OK" if res["ok"] else "FAILED",
                                 res["n_sums"], res["n_members"],
                                 res["n_verified"]),
        "",
        "- tarball: `%s` (%d bytes)" % (res["tar"], res["tar_bytes"]),
        "- computed SHA256: `%s`" % res["tar_sha256"],
        "- declared SHA256: `%s`" % res["declared_tar_sha256"],
    ]
    for key in ("missing", "extra", "mismatch"):
        if res[key]:
            lines.append("- %s: %d (%s)" % (key, len(res[key]),
                                            ", ".join(res[key][:5])))
    if res["problems"]:
        lines += ["", "## Problems", ""] + ["- " + x for x in res["problems"]]
    lines += ["", "Verified in memory; nothing was extracted to disk.", ""]
    return "\n".join(lines)
