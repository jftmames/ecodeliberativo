# mnl.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

def fit_mnl(df: pd.DataFrame, features: list[str]):
    """
    Ajusta un modelo Multinomial Logit (MNLogit) sobre df["Y"].
    """
    X = sm.add_constant(df[features], has_constant="add")
    y = df["Y"]
    model = MNLogit(y, X).fit(disp=False)
    return model

def predict_mnl(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Dado un modelo MNLogit y un df con las mismas features,
    devuelve un DataFrame con las probabilidades de cada categoría de Y.
    """
    X_new = sm.add_constant(df[features], has_constant="add")
    # predict devuelve un numpy array (n_obs x n_categorías)
    probs = model.predict(X_new)
    # Extraemos las categorías únicas de Y
    cats = np.unique(model.model.endog)
    # Nombramos las columnas como P(Y=cat)
    col_names = [f"P(Y={cat})" for cat in cats]
    # Construimos el DataFrame manteniendo el mismo índice que df
    probs_df = pd.DataFrame(probs, columns=col_names, index=df.index)
    return probs_df
