# mnl.py

import pandas as pd
import numpy as np
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
    # Obtenemos la matriz de probabilidades (n_obs × n_categories)
    probs = model.predict(X)

    # Determinar las categorías a partir del endogénico usado en el modelo
    # model.model.endog es un array con valores de Y
    categories = np.unique(model.model.endog)
    # Crear nombres para las columnas
    col_names = [f"P(Y={cat})" for cat in categories]

    # Finalmente devolvemos el DataFrame con las probabilidades y nombres adecuados
    return pd.DataFrame(probs, columns=col_names)
