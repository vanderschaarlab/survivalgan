# stdlib
from pathlib import Path
from typing import Tuple

# third party
import numpy as np
import pandas as pd
from lifelines.datasets import load_rossi
from sklearn.preprocessing import LabelEncoder
from sksurv.datasets import load_aids

try:
    from medicaldata.CUTRACT import download as cutract_download
    from medicaldata.CUTRACT import load as cutract_load
    from medicaldata.MAGGIC import download as maggic_download
    from medicaldata.MAGGIC import load as maggic_load
    from medicaldata.NCRAS import download as ncras_download
    from medicaldata.NCRAS import load as ncras_load
    from medicaldata.SEER_prostate_cancer import download as seer_download
    from medicaldata.SEER_prostate_cancer import load as seer_load
except BaseException:
    pass


def get_dataset(name: str) -> Tuple[pd.DataFrame, str, str, list]:
    data_folder = Path("data")
    data_folder.mkdir(parents=True, exist_ok=True)

    if name == "metabric":
        raw_df = pd.read_csv("data/metabric.csv")
        X = raw_df.drop(columns=["overall_survival_months", "overall_survival"])
        T = raw_df["overall_survival_months"]
        E = raw_df["overall_survival"]

        df = X.copy()
        df["event"] = E
        df["duration"] = T
    elif name == "rossi":
        df = load_rossi()
        df = df.rename(columns={"week": "duration", "arrest": "event"})
    elif name == "aids":
        X, Y = load_aids()
        Y_unp = np.array(Y, dtype=[("event", "int"), ("duration", "float")])
        df = X.copy()
        df["event"] = Y_unp["event"]
        df["duration"] = Y_unp["duration"]
    elif name == "cutract":
        file_id = "1mew1S3-N2GdVu5nGjaqmpLo7sTKRf4Vj"
        csv_path = data_folder / "cutract.csv"
        if not csv_path.exists():
            cutract_download(file_id, csv_path)

        X, T, Y = cutract_load(csv_path, preprocess=False)
        df = X.copy()
        df["event"] = Y
        df["duration"] = T

    elif name == "seer":
        file_id = "1PNXLjy8r1xHZq7SspduAMK6SGUTvuwM6"

        csv_path = data_folder / "seer.csv"
        if not csv_path.exists():
            seer_download(file_id, csv_path)

        X, T, Y = seer_load(csv_path, preprocess=False)
        df = X.copy()
        df["event"] = Y
        df["duration"] = T
    elif name == "maggic" or name == "pheart":
        file_id = "19Zvlxid9apEfI6dxuIrygOJCpckQVnX3"

        csv_path = data_folder / "maggic.csv"
        if not csv_path.exists():
            maggic_download(file_id, csv_path)

        X, T, Y = maggic_load(csv_path, preprocess=False)
        df = X.copy()
        df["event"] = Y
        df["duration"] = T
    elif name == "ncras_bc":
        file_id = "13BiNTv4koBfYAqKbXuQ2ORKNBelQjAYx"

        csv_path = data_folder / "ncras_bc.csv"
        if not csv_path.exists():
            ncras_download(file_id, csv_path)

        X, T, Y = ncras_load(csv_path, preprocess=False)
        df = X.copy()
        df["event"] = Y
        df["duration"] = T

    for col in df.columns:
        if df[col].dtype.name in ["object", "category"]:
            df[col] = LabelEncoder().fit_transform(df[col])

    duration_col = "duration"
    event_col = "event"

    df = df.fillna(0)

    T = df[duration_col]

    time_horizons = np.linspace(T.min(), T.max(), num=5)[1:-1].tolist()

    return df, duration_col, event_col, time_horizons
