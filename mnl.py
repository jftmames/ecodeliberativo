# mnl.py

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
import pandas as pd
import numpy as np

def fit_mnl(df: pd.DataFrame, features: list, target: str):
    X = sm.add_constant(df[features])
    y = df[target]
    model = MNLogit(y, X).fit(disp=False)
    return model

def summarize_mnl(model) -> str:
    return model.summary().as_text()

def predict_mnl(model, df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Calcula las probabilidades de cada alternativa para cada observación.
    Ahora genera columnas genéricas alt_0, alt_1, ..., alt_{K-1}.
    """
    X = sm.add_constant(df[features])
    probs = model.predict(X)  # numpy array of shape (n_obs, n_choices)
    probs = np.asarray(probs)
    n_choices = probs.shape[1]
    col_names = [f"alt_{i}" for i in range(n_choices)]
    return pd.DataFrame(probs, columns=col_names)

def chosen_alternative(probs_df: pd.DataFrame) -> pd.Series:
    return probs_df.idxmax(axis=1)

