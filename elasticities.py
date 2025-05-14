# elasticities.py

import pandas as pd
import numpy as np
import statsmodels.api as sm

def compute_elasticities(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Calcula elasticidades para un modelo Logit estimado con Statsmodels.
    Fórmula aproximada de elasticidad media:
        E_j = β_j * x̄_j * p̄ * (1 - p̄)
    donde x̄_j es la media de la variable j y p̄ la probabilidad media predicha.
    """
    # Reconstruir la matriz de diseño para predecir probabilidades
    X = sm.add_constant(df[features])
    # Probabilidades medias
    p = model.predict(X)
    p_mean = np.mean(p)
    # Extraer betas (excluimos el intercepto)
    betas = model.params[1:].to_dict()  # {'feature': beta, ...}
    # Calcular elasticidades
    elas = []
    for feat in features:
        beta_j = betas.get(feat, 0.0)
        x_mean = df[feat].mean()
        # Elasticidad
        E_j = beta_j * x_mean * p_mean * (1 - p_mean)
        elas.append({"Variable": feat, "Elasticidad": E_j})
    return pd.DataFrame(elas)
