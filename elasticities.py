# elasticities.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List

def compute_elasticities(model, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    Calcula elasticidades puntuales promedio para un modelo Logit.
    
    Para cada variable x_k:
      E_i,k = β_k * x_{i,k} * p_i * (1 - p_i)
    y se promedia sobre i.
    
    Devuelve un DataFrame con columnas ["Variable", "Elasticidad"].
    """
    # Reconstruimos la matriz de exógenas con constante
    X = sm.add_constant(df[features], has_constant="add")
    # Probabilidades ajustadas p_i
    p = model.predict(X)
    # Coeficientes β_k (excluimos const)
    params = model.params

    elasticities = []
    for k in features:
        beta_k = params.get(k, 0.0)
        x_k = df[k].astype(float)
        # Elasticidad puntual
        e_ik = beta_k * x_k * p * (1 - p)
        # Elasticidad promedio
        E_k = e_ik.mean()
        elasticities.append({"Variable": k, "Elasticidad": E_k})

    return pd.DataFrame(elasticities)
