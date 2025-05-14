# elasticities.py

import numpy as np
import pandas as pd

def compute_elasticities(model, df: pd.DataFrame, features: list[str]):
    """
    Para un modelo Logit, calcula elasticidades puntuales:
      ε_j = β_j * x_j * p * (1 − p)
    donde p = P(Y=1|X) y β_j es el coeficiente de feature j.

    Devuelve un DataFrame con columnas:
      - Variable
      - Elasticidad
    """
    coefs = model.params
    X = model.model.exog  # incluye la constante en posición 0
    p = model.predict(X)
    elas = []

    for idx, feat in enumerate(['const'] + features):
        if feat == 'const':
            continue
        β = coefs[feat]
        xj = df[feat]
        # Elasticidad promedio en la muestra
        ε = np.mean(β * xj * p * (1 - p))
        elas.append({'Variable': feat, 'Elasticidad': ε})

    return pd.DataFrame(elas)
