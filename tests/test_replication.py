"""Checks for replication_ds008836.py: the pre-registered decision rule on
synthetic feature tables. Run: python tests/test_replication.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replication_ds008836 import analyses, verdict  # noqa: E402

rng = np.random.default_rng(7)


def synthetic(slope: float) -> pd.DataFrame:
    rows = []
    for s in range(30):
        scale = rng.uniform(0.01, 0.06)  # people differ in raw ratio
        for ses, block in (("ses-1", "silent_probe"), ("ses-2", "silent_emerge")):
            for k in range(15):
                z = rng.standard_normal()
                depth = np.clip(np.round(3 + slope * z + rng.normal(0, 0.9)), 1, 5)
                rows.append({
                    "subject": f"sub-{s:03d}", "session": ses, "block": block,
                    "block_type": "silent", "variant": block.split("_")[1],
                    "probe_onset_s": 60.0 * k, "depth_rating": depth,
                    "minimal_ratio": scale * (1 + 0.2 * z),
                    "minimal_gamma_power": scale * (1 + 0.2 * z),
                    "minimal_theta_power": rng.uniform(1, 2),
                    "left_temporal_gamma_log": rng.normal(),
                    "right_temporal_gamma_log": rng.normal()})
    return pd.DataFrame(rows)


everyone = {f"sub-{s:03d}" for s in range(30)}

def run(slope: float) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    out = analyses(synthetic(slope=slope), everyone)
    return out.iloc[0], out.iloc[1], out


# Planted negative slope: the repaired primary test is valid, significant,
# and the per-participant cross-check agrees -> replicated.
primary, cross, out = run(-0.5)
assert primary["fit_valid"] and primary["estimate"] < 0, primary
assert primary["p_one_sided_negative"] < 0.05 and cross["p_one_sided_negative"] < 0.05
assert verdict(primary, cross).startswith("REPLICATED")
assert "of 30 participants negative" in out.iloc[-1]["note"]
# The original pre-registered mixed model is still reported, labelled as such.
assert out.iloc[2]["method"] == "mixed model as pre-registered"

# No signal, or the wrong direction: never replicated (one-sided test).
for slope in (0.0, +0.5):
    primary, cross, _ = run(slope)
    assert not verdict(primary, cross).startswith("REPLICATED"), (slope, primary)

# Decision rule: disagreement -> inconclusive; invalid fit -> inconclusive.
ok = pd.Series({"fit_valid": True, "estimate": -0.5, "p_one_sided_negative": 0.01})
no = pd.Series({"fit_valid": True, "estimate": -0.1, "p_one_sided_negative": 0.30})
assert verdict(ok, ok).startswith("REPLICATED")
assert verdict(ok, no).startswith("INCONCLUSIVE")
assert verdict(no, ok).startswith("INCONCLUSIVE")
assert verdict(no, no).startswith("NOT")
bad = ok.copy(); bad["fit_valid"] = False
assert verdict(bad, ok).startswith("INCONCLUSIVE")

print("ok: replication")
