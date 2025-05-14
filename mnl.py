# mnl.py

import pandas as pd
import statsmodels.api as sm
import numpy as np

def fit_mnl(df: pd.DataFrame, features: list[str], endog_name: str = "Y"):
    """
    Ajusta un modelo MNLogit.
    """
    X = sm.add_constant(df[features], has_constant="add")
    y = df[endog_name]
    model = sm.MNLogit(y, X).fit(disp=False)
    return model

def predict_mnl(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Devuelve DataFrame n_obs × n_cats con las probabilidades P(Y=cat).
    """
    # Reconstruye X como array NumPy
    X = sm.add_constant(df[features], has_constant="add")
    X_mat = X.values  # <— aquí el truco

    # statsmodels MNLogit.predict admite ahora un array limpio
    probs = model.predict(X_mat)  # shape (n_obs, n_cats) numpy array

    # Extrae categorías únicas ordenadas
    cats = np.unique(model.model.endog)
    cats.sort()

    # Nombra columnas
    col_names = [f"P(Y={cat})" for cat in cats]

    # Construye DataFrame con mismo índice
    probs_df = pd.DataFrame(probs, columns=col_names, index=df.index).astype(float)

    return probs_df
