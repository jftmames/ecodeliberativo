# mnl.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

def fit_mnl(df: pd.DataFrame, features: list[str]) -> MNLogit:
    """
    Ajusta un modelo de elección multinomial (MNLogit) usando statsmodels.
    - df: DataFrame que incluye la variable objetivo "Y" categórica.
    - features: lista de nombres de columnas explicativas.
    Devuelve el objeto ajustado MNLogit.
    """
    # Matriz de diseño con constante
    X = sm.add_constant(df[features], has_constant='add')
    y = df["Y"]
    model = MNLogit(y, X).fit(disp=False)
    return model

def predict_mnl(model: MNLogit, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Dados un modelo MNLogit ajustado y un DataFrame, predice probabilidades
    para cada categoría de Y.
    - model: objeto MNLogit ajustado.
    - df: DataFrame con las mismas features usadas en el ajuste.
    - features: lista de nombres de columnas explicativas.
    Retorna un DataFrame con columnas P(Y=cat) para cada categoría presente en Y.
    """
    # Reconstruimos X
    X_new = sm.add_constant(df[features], has_constant='add')
    # shape = (n_obs, n_categories)
    prob_array = model.predict(X_new)
    # Extraemos las categorías únicas en el mismo orden que aparece en el endógeno
    cats = list(dict.fromkeys(model.model.endog))  # preserva orden de aparición
    # Formamos nombres de columna
    col_names = [f"P(Y={cat})" for cat in cats]
    # Devolvemos DataFrame
    return pd.DataFrame(prob_array, columns=col_names, index=df.index)
