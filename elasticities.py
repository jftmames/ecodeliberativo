# elasticities.py

import pandas as pd

def compute_elasticities(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Calcula elasticidades puntuales al promedio de las variables en un Logit.
    Elasticidad_j = β_j * p_mean * (1 - p_mean) * (mean(x_j) / p_mean)
                   = β_j * (1 - p_mean) * mean(x_j)
    """
    X = model.model.exog  # matriz con constante + features
    p = model.predict(X)  # vector de probabilidades
    p_mean = p.mean()
    results = []
    for feat in features:
        beta_j = model.params[feat]
        x_mean = df[feat].mean()
        elas = beta_j * (1 - p_mean) * x_mean
        results.append((feat, elas))
    return pd.DataFrame(results, columns=["Variable", "Elasticidad"])
