# stdlib
import string
import warnings
from pathlib import Path

# third party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from lifelines import KaplanMeierFitter
from sklearn.manifold import TSNE
# synthcity absolute
from synthcity.benchmark import Benchmarks
from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import SurvivalAnalysisDataLoader
from synthcity.plugins.core.models.survival_analysis.metrics import \
    nonparametric_distance
from synthcity.utils.serialization import (dataframe_hash, load_from_file,
                                           save_to_file)

from datasets import get_dataset

warnings.filterwarnings("ignore")

out_dir = Path("output")
workspace = Path("workspacei_rebuttal")

fontsize = 14
plt.style.use("seaborn-whitegrid")


gain_scenarios = [
    (
        "w/o TTE Regressor",
        {
            "uncensoring_model": "date",
            "tte_strategy": "survival_function",
            "dataloader_sampling_strategy": "none",
        },
    ),
    (
        "w/o Imbalanced Sampling",
        {
            "uncensoring_model": "survival_function_regression",
            "tte_strategy": "survival_function",
            "dataloader_sampling_strategy": "none",
        },
    ),
    (
        "w/o Temporal Sampling",
        {
            "uncensoring_model": "survival_function_regression",
            "tte_strategy": "survival_function",
            "dataloader_sampling_strategy": "imbalanced_censoring",
        },
    ),
    (
        "w/o Cond. GAN ",
        {
            "uncensoring_model": "survival_function_regression",
            "tte_strategy": "survival_function",
            "dataloader_sampling_strategy": "imbalanced_time_censoring",
        },
    ),
]


def evaluate_dataset(dataset: str, scenarios: list):
    df, duration_col, event_col, time_horizons = get_dataset(dataset)
    # experiment = "gain_of_function_parametric"
    dataloader = SurvivalAnalysisDataLoader(
        df,
        target_column=event_col,
        time_to_event_column=duration_col,
        time_horizons=time_horizons,
    )

    experiment = "sources_of_gain_parametric"
    for scenario_name, scenario_args in scenarios:
        bkp = out_dir / f"experiment_{experiment}_{dataset}_{scenario_name}.bkp"

        if bkp.exists():
            score = load_from_file(bkp)
        else:
            score = Benchmarks.evaluate(
                [(scenario_name, "survae", scenario_args)],
                dataloader,
                task_type="survival_analysis",
                repeats=2,
                metrics={"performance": ["linear_model", "xgb"]},
                workspace=workspace,
            )

            save_to_file(bkp, score)

        print("Scenario", scenario_name, scenario_args)
        Benchmarks.print(score)


evaluate_dataset("metabric", gain_scenarios)
