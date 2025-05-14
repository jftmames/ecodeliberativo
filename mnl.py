# mnl.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
from typing import List, Union

def fit_mnl(df: pd.DataFrame, features: List[str]) -> sm.MNLogitResults:
    """
    Ajusta un modelo Multinomial Logit (MNL) sobre df.
    - df: DataFrame que incluye la variable Y categórica.
    - features: lista de nombres de columnas explicativas.
    Retorna el objeto resultados de MNLogit.
    """
    # Diseño de la matriz de regresores con constante
    X = sm.add_constant(df[features], has_constant="add")
    y = df["Y"]
    # Ajuste sin mostrar salida en consola
    model = MNLogit(y, X).fit(disp=False)
    return model

def predict_mnl(model: sm.MNLogitResults,
                df: pd.DataFrame,
                features: List[str]
               ) -> pd.DataFrame:
    """
    Predice las probabilidades de cada categoría de Y para las filas de df.
    - model: objeto retornado por fit_mnl.
    - df: DataFrame con las mismas features usadas en el ajuste.
    - features: lista de nombres de columnas explicativas.
    Retorna un DataFrame de probabilidades, con una columna por categoría de Y.
    """
    # Construir diseño idéntico al de entrenamiento
    X = sm.add_constant(df[features], has_constant="add")
    # predict devuelve un array (n_obs, n_categories)
    probs = model.predict(X)
    # Obtener etiqutas de las categorías de Y, ordenadas
    # model.model.endog es el vector original de Y
    cats = sorted(pd.unique(model.model.endog))
    # Convertir a DataFrame con nombres de columnas iguales a las categorías
    df_probs = pd.DataFrame(probs, columns=[str(c) for c in cats], index=df.index)
    return df_probs
