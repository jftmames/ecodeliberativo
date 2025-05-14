# count_models.py

import statsmodels.api as sm
import pandas as pd

def fit_poisson(df: pd.DataFrame, features: list[str]):
    """
    Ajusta un modelo de Poisson:
    - df: DataFrame con Y de conteo y las features.
    - features: lista de nombres de columna en df usadas como explicativas.
    Devuelve un PoissonResults.
    """
    X = sm.add_constant(df[features], has_constant='add')
    y = df['Y']
    model = sm.Poisson(y, X).fit(disp=False)
    return model

def fit_nb(df: pd.DataFrame, features: list[str]):
    """
    Ajusta un modelo Neg Binomial:
    - df: DataFrame con Y de conteo y las features.
    - features: lista de nombres de columna en df usadas como explicativas.
    Devuelve un NegativeBinomialResults.
    """
    X = sm.add_constant(df[features], has_constant='add')
    y = df['Y']
    model = sm.NegativeBinomial(y, X).fit(disp=False)
    return model

def predict_count(model, df: pd.DataFrame, features: list[str]):
    """
    Dado un PoissonResults o NegativeBinomialResults y un nuevo df+features,
    devuelve un DataFrame con la esperanza E[Y|X].
    """
    X = sm.add_constant(df[features], has_constant='add')
    mu = model.predict(X)
    return pd.DataFrame({'E[Y]': mu})
