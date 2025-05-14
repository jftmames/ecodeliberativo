# mnl.py

import pandas as pd
import statsmodels.api as sm

def fit_mnl(df: pd.DataFrame, features: list[str]):
    """
    Ajusta un modelo Multinomial Logit (MNLogit) usando Statsmodels.
    """
    X = sm.add_constant(df[features])
    y = df["Y"]
    model = sm.MNLogit(y, X).fit(disp=False)
    return model

def predict_mnl(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Predice probabilidades para cada categoría de Y con un modelo MNLogit.
    Devuelve un DataFrame con columnas P(Y=cat).
    """
    X = sm.add_constant(df[features], has_constant="add")
    probs = model.predict(X)  # shape (n_obs, n_cats)
    # Obtener categorías únicas en el orden que usó el modelo
    cats = model.model.endog_unique
    col_names = [f"P(Y={cat})" for cat in cats]
    return pd.DataFrame(probs, columns=col_names)
