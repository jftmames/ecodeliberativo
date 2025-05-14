# mnl.py

import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

def fit_mnl(df: pd.DataFrame, features: list[str]):
    """
    Ajusta un modelo de elección múltiple (MNLogit) sobre df,
    usando las columnas `features` como regresoras y 'Y' como variable dependiente.
    """
    # Añadimos constante
    X = sm.add_constant(df[features], has_constant="add")
    y = df["Y"]
    model = MNLogit(y, X).fit(disp=False)
    return model

def predict_mnl(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Dada la instancias en df (solo con las columnas `features`),
    añade constante y devuelve un DataFrame con las probabilidades
    de cada alternativa según el modelo MNLogit ajustado.
    """
    # Asegurarnos de que la constante está incluida
    X = sm.add_constant(df[features], has_constant="add")
    # model.predict espera un array 2D, con misma orden de columnas que en fit
    probs = model.predict(X)
    # `model.model.endog_names` contiene los nombres de las alternativas
    return pd.DataFrame(probs, columns=model.model.endog_names)
