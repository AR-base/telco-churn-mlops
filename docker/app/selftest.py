"""Checks the image before it is used: `python selftest.py` (also run automatically during `docker build`).

1. the model file matches the checksum recorded when it was exported from the MLflow registry;
2. it loads with only the allowed types (skops, no pickle);
3. it gives exactly the recorded probabilities for 25 real test customers;
4. the API's input rules still reject an impossible customer;
5. the website and what-if code compile.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MODEL_DIR", str(HERE / "model"))
sys.path.insert(0, str(HERE))
results = []


def check(name, ok):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name, flush=True)


from churn_api import Customer, load_exported                      # noqa: E402
from churn_features import PREDICTORS                             # noqa: E402

try:
    pipeline, info = load_exported(os.environ["MODEL_DIR"])
    check("model file matches its export checksum and loads with the allowed types only", True)
except Exception as e:                                             # report, then fail below
    check(f"model loads ({type(e).__name__}: {e})", False)
    sys.exit(1)

cases = info["selftest"]
X = pd.DataFrame([Customer(**c).to_row() for c in cases["customers"]])[PREDICTORS]
got = pipeline.predict_proba(X)[:, 1]
check(f"same probabilities as in the notebook for {len(got)} test customers "
      f"(max difference {np.abs(got - np.array(cases['probabilities'])).max():.1e})",
      np.allclose(got, cases["probabilities"], atol=1e-9))

bad = dict(cases["customers"][0], InternetService="No")            # add-ons would still say Yes/No
try:
    Customer(**bad)
    check("impossible customer is rejected", False)
except ValueError:
    check("impossible customer is rejected", True)

for f in ("churn_app.py", "whatif.py", "start.py"):
    compile((HERE / f).read_text(), f, "exec")                    # syntax check only, writes nothing
check("website, what-if and entrypoint code compile", True)

print(f"\n{sum(results)}/{len(results)} checks passed — model {info['name']} v{info['version']} "
      f"({info['config']}, run {info['run_id'][:8]})", flush=True)
sys.exit(0 if all(results) else 1)
