import pandas as pd
import numpy as np

def compute_elasticities(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Calcula elasticidades puntuales para un modelo logit binario:
      e_j = β_j * x_j * (1 - p)
    donde p = P(Y=1|X) en el punto medio de los datos.
    Devuelve un DataFrame con columnas ['Variable', 'Elasticidad'].
    """
    # Punto de evaluación: media de X
    x0 = df[features].mean()
    X0 = [1.0] + list(x0)
    p0 = model.predict([X0])[0]
    betas = model.params.values

    rows = []
    # Ignoramos β0
    for j, feat in enumerate(features, start=1):
        βj = betas[j]
        xj = x0[feat]
        elasticidad = βj * xj * (1 - p0)
        rows.append({"Variable": feat, "Elasticidad": elasticidad})

    return pd.DataFrame(rows)
