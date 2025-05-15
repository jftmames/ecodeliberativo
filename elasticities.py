# elasticities.py

import pandas as pd
import numpy as np

def compute_elasticities(model, df, features):
    """
    Calcula elasticidades marginales para modelos lineales y logit (apróx.).
    Para lineales: β·x̄/Ȳ
    Para Logit: β·x̄·(1-p̄) si Y binario
    Para Poisson: β·x̄
    Devuelve DataFrame: Variable | Elasticidad
    """
    # Cálculo general para lineal (OLS)
    ybar = df["Y"].mean()
    elasticities = []
    params = model.params

    # Probabilidad media, para modelos tipo logit/probit
    try:
        pbar = model.predict(df[features]).mean()
    except Exception:
        pbar = np.nan

    for i, feat in enumerate(features, 1):  # [1:] para saltar la constante
        beta = params[feat] if feat in params else params[i]
        xbar = df[feat].mean()

        if hasattr(model, "model"):
            mclass = type(model.model).__name__
        else:
            mclass = type(model).__name__

        # Modelo OLS
        if "OLS" in mclass:
            elas = beta * xbar / ybar if ybar else np.nan
        # Modelo Logit
        elif "Logit" in mclass:
            elas = beta * xbar * (1 - pbar) if not np.isnan(pbar) else np.nan
        # Modelo Probit
        elif "Probit" in mclass:
            elas = beta * xbar * (1 - pbar) if not np.isnan(pbar) else np.nan
        # Modelo Poisson
        elif "Poisson" in mclass:
            elas = beta * xbar
        # Otros
        else:
            elas = np.nan

        elasticities.append({"Variable": feat, "Elasticidad": elas})

    return pd.DataFrame(elasticities)
