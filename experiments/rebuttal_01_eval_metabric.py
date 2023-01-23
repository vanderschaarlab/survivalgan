# stdlib
import sys
import warnings
from pathlib import Path

# third party
import pandas as pd
# synthcity absolute
import synthcity.logger as log
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.utils.serialization import load_from_file, save_to_file

from datasets import get_dataset

out_dir = Path("output")
bkp = out_dir / f"metrics.metabric.bkp"


df, duration_col, event_col, time_horizons = get_dataset("metabric")
X = df.drop(columns=[duration_col, event_col])
T = df[duration_col]
E = df[event_col]
df

# third party
from adjutorium.plugins.prediction.risk_estimation import RiskEstimation

model = RiskEstimation().get("survival_xgboost")

model.fit(X, T, E)

# third party
from adjutorium.plugins.explainers import Explainers

exp = Explainers().get(
    "risk_effect_size",
    model,
    X,
    E,
    time_to_event=T,
    eval_times=time_horizons,
    task_type="risk_estimation",
    effect_size=0.5,
)
exp.explain(X).index

value_of_inf = exp.explain(X).index.tolist()
important_features = value_of_inf

Xeval = X[important_features]
df_eval = Xeval.copy()
df_eval[duration_col] = T
df_eval[event_col] = E

dataloader = SurvivalAnalysisDataLoader(
    df,
    target_column=event_col,
    time_to_event_column=duration_col,
    time_horizons=time_horizons,
    important_features=important_features[:20],
)


log.remove()
log.add(sink=sys.stderr, level="DEBUG")

scores = Benchmarks.evaluate(
    [
        # ("ctgan", "ctgan", {}),
        # ("tvae", "tvae", {}),
        # ("survival_gan", "survival_gan", {}),
        ("survival_vae", "survae", {}),
        # ("nflow", "nflow", {}),
        # ("privbayes", "privbayes", {}),
    ],
    dataloader,
    task_type="survival_analysis",
    repeats=3,
    metrics={"performance": ["linear_model", "xgb"]},
    # metrics = {},
    synthetic_size=10 * len(df),
    workspace="workspace_rebuttal_10x",
)

Benchmarks.print(scores)
save_to_file(bkp, scores)
