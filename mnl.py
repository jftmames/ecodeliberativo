# mnl.py

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
import pandas as pd

def fit_mnl(df: pd.DataFrame, features: list, target: str):
    """
    Ajusta un modelo Multinomial Logit:
      P(y = k | X) = exp(X·β_k) / Σ_j exp(X·β_j)
    Parámetros:
      - df: DataFrame con los datos
      - features: lista de nombres de columnas (regresores)
      - target: nombre de la columna de la variable dependiente (categoría)
    Devuelve:
      - model: objeto resultante de MNLogit.fit()
    """
    X = sm.add_constant(df[features])
    y = df[target]
    model = MNLogit(y, X).fit(disp=False)
    return model

def summarize_mnl(model) -> str:
    """
    Devuelve el summary() del modelo MNL como texto.
    """
    return model.summary().as_text()

def predict_mnl(model, df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Calcula las probabilidades de cada alternativa para cada observación.
    Devuelve un DataFrame con columnas para cada categoría.
    """
    X = sm.add_constant(df[features])
    probs = model.predict(X)
    # statsmodels devuelve un numpy array; convertimos a DataFrame
    return pd.DataFrame(probs, columns=model.model.endog_names)

def chosen_alternative(probs_df: pd.DataFrame) -> pd.Series:
    """
    Dada la matriz de probabilidades, devuelve la alternativa con mayor probabilidad.
    """
    return probs_df.idxmax(axis=1)
