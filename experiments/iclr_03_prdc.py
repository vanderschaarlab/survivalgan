import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import synthcity.logger as log

from datasets import get_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

log.remove()
warnings.filterwarnings("ignore")
log.add(sink=sys.stderr, level="DEBUG")

from pathlib import Path

import tabulate
from adjutorium.utils.metrics import generate_score, print_score
from synthcity.metrics.eval_statistical import PRDCScore
from synthcity.plugins.core.dataloader import (GenericDataLoader,
                                               SurvivalAnalysisDataLoader)
from synthcity.utils.serialization import (dataframe_hash, load_from_file,
                                           save_to_file)

log.remove()

datasets = [
    "aids",
    "cutract",
    "maggic",
    "seer",
]
methods = ["survival_gan", "ctgan", "nflow", "tvae", "privbayes", "adsgan"]

out_dir = Path("workspace_rebuttal")
headers = ["dataset"] + methods


def evaluate_metric(metric: str):
    distances = []
    for ref_df in ["maggic"]:

        df, duration_col, event_col, time_horizons = get_dataset(ref_df)
        df_hash = dataframe_hash(df)

        real_dataloader = GenericDataLoader(
            df,
        )
        local_distance = [ref_df]
        for method in methods:
            scores = []
            for seed in range(3):
                model_bkp = out_dir / f"{df_hash}_{method}_{seed}.bkp"
                syn_df = load_from_file(model_bkp)
                try:
                    syn_df = syn_df.dataframe()
                except BaseException as e:
                    pass

                syn_dataloader = GenericDataLoader(
                    syn_df,
                )
                if len(syn_df) == 0:
                    continue

                score = PRDCScore().evaluate(real_dataloader, syn_dataloader)[
                    f"{metric}_OC"
                ]
                scores.append(score)
            final_score = print_score(generate_score(scores))

            local_distance.append(final_score)
        distances.append(local_distance)

    print("=================================================")
    print("RESULTS:", metric)
    print(tabulate.tabulate(distances, headers=headers))


evaluate_metric("precision")
evaluate_metric("recall")
evaluate_metric("coverage")
evaluate_metric("density")
