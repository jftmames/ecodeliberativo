# probit_tobit.py

import statsmodels.api as sm
import pandas as pd

def fit_probit(df: pd.DataFrame, features: list[str]):
    """
    Ajusta un modelo Probit:
    - df: DataFrame con Y binaria y las features.
    - features: lista de nombres de columna en df usadas como explicativas.
    Devuelve un objeto ProbitResults.
    """
    X = sm.add_constant(df[features], has_constant='add')
    y = df['Y']
    model = sm.Probit(y, X).fit(disp=False)
    return model


def predict_probit(model, df: pd.DataFrame, features: list[str]):
    """
    Dado un ProbitResults y un nuevo df+features, devuelve
    un DataFrame con columnas:
      - P(Y=1)
      - P(Y=0)
    """
    X = sm.add_constant(df[features], has_constant='add')
    p1 = model.predict(X)           # Probabilidad de Y=1
    p0 = 1 - p1                     # Complemento
    return pd.DataFrame({
        'P(Y=1)': p1,
        'P(Y=0)': p0
    })


def fit_tobit(df: pd.DataFrame, features: list[str], left=0, right=None):
    """
    Ajusta un modelo Tobit (censurado).
    - left: límite inferior de censoring (por defecto 0).
    - right: límite superior (None = no hay).
    Devuelve un TobitResults.
    """
    from statsmodels.miscmodels.tobit import Tobit
    X = sm.add_constant(df[features], has_constant='add')
    y = df['Y']
    model = Tobit(y, X, left=left, right=right).fit(disp=False)
    return model


def predict_tobit(model, df: pd.DataFrame, features: list[str]):
    """
    Dado un TobitResults y un nuevo df+features, devuelve
    un DataFrame con la predicción de E[Y|X].
    """
    X = sm.add_constant(df[features], has_constant='add')
    preds = model.predict(X)
    return pd.DataFrame({'E[Y]': preds})
