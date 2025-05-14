import pandas as pd
import numpy as np

def compute_elasticities(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Calcula elasticidades puntuales promedio para un modelo Logit:
      Elasticidad_j = β_j * x̄_j * p̄ * (1 - p̄)
    donde x̄_j es la media de la variable j y p̄ la probabilidad media.
    """
    # predicciones y probabilidad media
    p = model.predict()
    p_bar = np.mean(p)
    rows = []
    for feat in features:
        beta_j = model.params.get(feat, 0.0)
        x_bar = df[feat].mean()
        elas = beta_j * x_bar * p_bar * (1 - p_bar)
        rows.append({"Variable": feat, "Elasticidad": elas})
    return pd.DataFrame(rows)
