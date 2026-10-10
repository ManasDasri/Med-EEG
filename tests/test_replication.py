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

out = analyses(synthetic(slope=-0.5), everyone)
primary = out.iloc[0]
assert primary["estimate"] < 0 and primary["p_one_sided_negative"] < 0.05, primary
assert verdict(primary).startswith("REPLICATED")
assert "of 30 participants negative" in out.iloc[-1]["note"]

null = analyses(synthetic(slope=0.0), everyone).iloc[0]
assert verdict(null).startswith("NOT"), null

wrong_way = analyses(synthetic(slope=+0.5), everyone).iloc[0]
assert wrong_way["estimate"] > 0 and verdict(wrong_way).startswith("NOT")  # one-sided

print("ok: replication")
