# stdlib
import sys
import warnings
from pathlib import Path

# third party
import pandas as pd
# synthcity absolute
import synthcity.logger as log
from medicaldata.SEER_prostate_cancer import download as seer_download
from medicaldata.SEER_prostate_cancer import load as seer_load
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.utils.serialization import load_from_file, save_to_file

from datasets import get_dataset

out_dir = Path("output")
bkp = out_dir / f"metrics.seer.bkp"


df, duration_col, event_col, time_horizons = get_dataset("seer")
X = df.drop(columns=[duration_col, event_col])
T = df[duration_col]
E = df[event_col]
df

dataloader = SurvivalAnalysisDataLoader(
    df,
    target_column=event_col,
    time_to_event_column=duration_col,
    time_horizons=time_horizons,
)


log.remove()
log.add(sink=sys.stderr, level="DEBUG")

scores = Benchmarks.evaluate(
    [
        (
            "survival_predicting_censoring",
            "survival_gan",
            {"censoring_strategy": "covariate_dependent"},
        ),
        # ("survival_gan", "survival_gan", {}),
        # ("survae", "survae", {}),
    ],
    dataloader,
    task_type="survival_analysis",
    repeats=3,
    metrics={"performance": ["linear_model", "xgb"]},
    # metrics = {},
    workspace="workspace_rebuttal",
    synthetic_reuse_if_exists=False,
)

Benchmarks.print(scores)
save_to_file(bkp, scores)
