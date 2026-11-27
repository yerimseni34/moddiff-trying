"""Offline end-to-end check (no API): mock 'style' = known 3D pattern, each view
randomly shifted/zoomed by up to 2%; the baked texture is compared to ground truth."""
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from texproj import pipeline as pl  # noqa: E402
from texproj.stylize import mock_color  # noqa: E402

tmp = Path(tempfile.mkdtemp())
model = tmp / "test.obj"
subprocess.check_call([sys.executable, str(Path(__file__).parent / "make_test_model.py"), str(model)])
work = pl.prepare(model, tmp / "work", views="6", size=768)
pl.stylize(work, [], backend="mock", jitter=0.02)
res = pl.project(work, dict(tex=1024, depth_res=1536))
b, hole = res["baker"], res["hole"]
got = res["texture"][b.valid].astype(np.float32) / 255
err = np.abs(got - mock_color(b.pos)).mean(1)
print(f"painted {1 - hole.mean():.1%} | mean abs error painted {err[~hole].mean():.4f}, "
      f"median {np.median(err[~hole]):.4f}, >0.1: {(err[~hole] > 0.1).mean():.1%}")
assert err[~hole].mean() < 0.03, "projection error too high"

# --- fix stage (mock AI paints the true pattern; only masked pixels may be used)
before_good = ~hole & (err < 0.05)
views = pl.fix_prepare(work, 4, dict(tex=1024, depth_res=1536))
assert views, "no fix views chosen"
pl.fix_stylize(work, [], backend="mock")
res2 = pl.project(work, dict(tex=1024, depth_res=1536))
got2 = res2["texture"][b.valid].astype(np.float32) / 255
err2 = np.abs(got2 - mock_color(b.pos)).mean(1)
print(f"after fix: painted {1 - res2['hole'].mean():.1%} (was {1 - hole.mean():.1%}) | "
      f"error on previously-good texels {err2[before_good].mean():.4f} (was {err[before_good].mean():.4f}) | "
      f"overall error {err2.mean():.4f} (was {err.mean():.4f})")
assert res2["hole"].mean() < hole.mean(), "fix did not paint anything new"
assert err2[before_good].mean() < err[before_good].mean() + 0.005, "fix damaged good paint"
assert err2.mean() < err.mean(), "fix did not improve the texture"
print("OK  ->", work / "out")
