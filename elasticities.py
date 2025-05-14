# elasticities.py

import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Logit
from typing import List

def compute_elasticities(
    model: Logit, 
    df: pd.DataFrame, 
    features: List[str]
) -> pd.DataFrame:
    """
    Calcula la elasticidad promedio de P(Y=1) respecto a cada variable continua.
    
    Elasticidad E_j = β_j * x_j * (1 - p)  [marginal multiplicativa]
    y luego se promedian sobre las observaciones.

    Parámetros:
    - model: objeto ajustado de statsmodels.discrete.discrete_model.LogitResults
    - df: DataFrame con los datos usados en el ajuste
    - features: lista de nombres de variables explicativas (deben ser continuas)

    Devuelve:
    DataFrame con columnas ["Variable", "Elasticidad"].
    """
    # Obtener betas
    params = model.params
    # Matriz X con constante
    X = model.model.exog
    # Predecir probabilidades
    p = model.predict(X)
    # Construir DataFrame de exógenas (coincidente con X sin constante)
    # statsmodels guarda nombres en model.exog_names
    exog_names = model.model.exog_names
    # Localizar índice de cada feature en exog_names
    idx = {name: exog_names.index(name) for name in exog_names}
    # Para cada feature, calcular E_j_i = β_j * x_{ij} * (1 - p_i)
    results = []
    for feat in features:
        if feat not in params.index:
            continue  # por si acaso
        beta_j = params[feat]
        # columna de valores x_j (usando X matriz)
        x_j = X[:, idx[feat]]
        # vector de elasticidades individuales
        ej_i = beta_j * x_j * (1 - p)
        # promedia sobre i
        E_j = np.mean(ej_i)
        results.append({"Variable": feat, "Elasticidad": E_j})

    return pd.DataFrame(results)
