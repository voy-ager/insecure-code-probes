"""Run the whole pipeline for one config, in order, resuming finished steps.

python scripts/run_all.py --config configs/pilot_0.5b.toml [--force] [--from STEP]

Steps run as subprocesses of this interpreter (our own scripts, no shell). Each finished step is
recorded with its wall-clock and peak RAM in <results>/run_log.json and skipped on the next run
unless the config, prompts file or pair templates changed, an expected output is gone, --force is given, or --from names an earlier step. The steps' own caches
(activations, generations) still apply when they re-run. Nothing is committed: the gate step only
says whether generation text may be committed (PII check from label_generations, plus gitleaks).
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess  # nosec B404 - runs this repo's own scripts with a fixed argv, never a shell
import sys
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)


def steps(config: str) -> list[tuple[str, list[str] | None]]:
    """(name, script argv); argv None = in-process step."""
    c = ["--config", config]
    return [
        ("build_pairs", ["scripts/build_pairs.py"]),
        ("extract_pairs", ["scripts/extract.py", *c, "--set", "pairs"]),
        ("generate", ["scripts/generate.py", *c]),
        ("label", ["scripts/label_generations.py", *c, "--spot-check", "20"]),
        ("gate", None),
        ("revision_check", ["scripts/revision_check.py", *c]),
        ("extract_transfer", ["scripts/extract.py", *c, "--set", "transfer"]),
        ("baselines", ["scripts/run_baselines.py", *c]),
        ("judge_pairs", ["scripts/judge.py", *c, "--set", "pairs"]),
        ("judge_transfer", ["scripts/judge.py", *c, "--set", "transfer"]),
        ("run_probes", ["scripts/run_probes.py", *c]),
        ("make_report", ["scripts/make_report.py", *c]),
        ("pilot_stats", ["scripts/revision_check.py", *c]),  # final stats incl. activations, judge, sanity
    ]


def outputs(cfg: dict) -> dict[str, list[Path]]:
    """Files a finished step must have left behind; a step is only 'done' while they all exist."""
    p = {k: Path(v) for k, v in cfg["paths"].items()}
    if not {"pairs", "activations", "generations", "results"} <= p.keys():
        return {}  # partial config (tests): no output checks
    return {
        "build_pairs": [p["pairs"]],
        "extract_pairs": [p["activations"] / "pairs_meta.json"],
        "generate": [p["generations"] / "generations.jsonl"],
        "label": [p["generations"] / "labels.jsonl"],
        "revision_check": [p["results"] / "pilot_stats.json"],
        "extract_transfer": [p["activations"] / "transfer_meta.json"],
        "baselines": [p["results"] / "scores_lexical_pairs.csv"],
        "judge_pairs": [p["results"] / "judge_meta.json"],
        "judge_transfer": [p["results"] / "judge_meta.json"],
        "run_probes": [p["results"] / "probe_layers.csv"],
    }


def inputs_sha(cfg: dict, config: str) -> str:
    """Hash of the config plus the inputs no step produces: prompts file and the pair templates."""
    h = hashlib.sha256(Path(config).read_bytes())
    paths = cfg["paths"]
    files = [Path(paths[k]) for k in ("prompts",) if k in paths]
    if "templates" in paths:
        files += sorted(Path(paths["templates"]).rglob("*"))
    for f in files:
        if f.is_file():
            h.update(f.as_posix().encode() + f.read_bytes())
    return h.hexdigest()


def peak_mb() -> float:
    """Peak RAM of the current process in MB (Windows peak working set; POSIX max RSS)."""
    if os.name != "nt":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB on Linux
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    c = Counters(cb=ctypes.sizeof(Counters))
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    get_info = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    get_info(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2**20


# The step measures its own peak RAM at exit: a venv's python.exe on Windows is a launcher that
# starts the real interpreter as a grandchild, so measuring the direct child would be wrong.
_CHILD = """
import atexit, runpy, sys
sys.path.insert(0, sys.argv[1])
import run_all
out = sys.argv[2]
def write_peak():
    with open(out, "w") as f:
        f.write(str(run_all.peak_mb()))
atexit.register(write_peak)
sys.argv = sys.argv[3:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def run_script(argv: list[str]) -> dict:
    """Run one script; returns seconds and peak RAM (MB). Raises on a non-zero exit."""
    t = time.perf_counter()
    if not (Path(__file__).parent.parent / argv[0]).exists():
        raise SystemExit(f"script not found: {argv[0]} (a dependency PR, e.g. #28 for label_generations.py, may not be merged)")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "peak"
        cmd = [sys.executable, "-c", _CHILD, str(Path(__file__).parent), str(out), *argv]
        rc = subprocess.run(cmd, check=False).returncode  # nosec B603 - fixed argv from steps(), no shell
        if rc:
            raise SystemExit(f"step failed (exit {rc}): {' '.join(argv)}")
        peak = round(float(out.read_text())) if out.exists() else None
    return {"seconds": round(time.perf_counter() - t, 1), "peak_ram_mb": peak}


def gate(cfg: dict) -> dict:
    """May generation text be committed? PII check (label step) and gitleaks on the generations dir."""
    d = Path(cfg["paths"]["generations"])
    if (d / "labeled.local.jsonl").exists():
        raise SystemExit("PII check flagged generations: commit labels.jsonl only; text stays in "
                         "labeled.local.jsonl. Review the flags, move labeled.local.jsonl out of "
                         f"{d}, then re-run with --from gate.")
    exe = shutil.which("gitleaks")
    if exe is None:
        log.warning("gitleaks not on PATH: run `gitleaks dir %s` before committing any generation text", d)
        return {"pii_clean": True, "gitleaks": "not run"}
    rc = subprocess.run([exe, "dir", str(d), "--redact", "--no-banner"], check=False).returncode  # nosec B603
    if rc:
        log.warning("gitleaks found leaks in %s: do not commit generation text", d)
    return {"pii_clean": True, "gitleaks": "fail" if rc else "pass"}


def main(argv=None, run=run_script) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="re-run every step")
    ap.add_argument("--from", dest="start", help="re-run from this step on")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # imported here, not at module level: the step wrapper imports this module without icp
    from icp import load_config

    cfg = load_config(args.config)
    plan = steps(args.config)
    names = [n for n, _ in plan]
    if args.start and args.start not in names:
        raise SystemExit(f"--from must be one of {names}")
    res = Path(cfg["paths"]["results"])
    res.mkdir(parents=True, exist_ok=True)
    log_path = res / "run_log.json"
    sha = inputs_sha(cfg, args.config)
    expected = outputs(cfg)
    state = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {}
    if args.force or state.get("config_sha256") != sha:
        state = {"config_sha256": sha, "steps": {}}
    redo = set(names[names.index(args.start):]) if args.start else set()
    for name, cmd in plan:
        if name in state["steps"] and name not in redo and all(f.exists() for f in expected.get(name, [])):
            log.info("skip %s (done)", name)
            continue
        log.info("step %s", name)
        rec = gate(cfg) if cmd is None else run(cmd)
        counts = Path(cfg["paths"]["generations"]) / "generate_run.json"
        if name == "generate" and counts.exists():
            rec.update(json.loads(counts.read_text(encoding="utf-8")))
        state["steps"][name] = rec
        log_path.write_text(json.dumps(state, indent=1), encoding="utf-8")  # after every step: resumable


if __name__ == "__main__":
    main()
