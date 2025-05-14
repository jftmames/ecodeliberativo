# econometrics.py
import statsmodels.api as sm
import pandas as pd

def add_constant(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Añade la constante al diseño X para regresión.
    """
    X = df[cols].copy()
    return sm.add_constant(X)

def fit_logit(df: pd.DataFrame, features: list, target: str):
    """
    Ajusta un modelo Logit y devuelve el objeto resultante.
    """
    X = add_constant(df, features)
    y = df[target]
    model = sm.Logit(y, X).fit(disp=False)
    return model

def fit_ols(df: pd.DataFrame, features: list, target: str):
    """
    Ajusta un modelo OLS y devuelve el objeto resultante.
    """
    X = add_constant(df, features)
    y = df[target]
    model = sm.OLS(y, X).fit()
    return model

def summarize_model(model) -> str:
    """
    Devuelve el summary() del modelo como texto.
    """
    return model.summary().as_text()

def compute_average_probability(model, df: pd.DataFrame, features: list) -> float:
    """
    Para Logit, calcula la probabilidad media de elección.
    """
    X = add_constant(df, features)
    probs = model.predict(X)
    return float(probs.mean())
