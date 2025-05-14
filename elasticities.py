import numpy as np
import pandas as pd
import statsmodels.api as sm


def compute_elasticities(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Calcula elasticidades puntuales para un modelo Logit o Logit Multinomial.

    Para el modelo Logit binario, la elasticidad se evalúa en la media de X:
        E_j = beta_j * x_j_mean * p_mean * (1 - p_mean)

    Para MNL, se computan elasticidades arc: aproximación local.

    model: objeto ajustado (Logit o MNLogit)
    df: DataFrame con datos originales
    features: lista de columnas independientes

    Retorna DataFrame con columnas ['variable', 'elasticidad']
    """
    # Preparar datos
    X = sm.add_constant(df[features])
    # Obtener probabilidades predichas
    try:
        p = model.predict(X)
        # Si es array 2D (MNL), tomar la primera clase
        if p.ndim > 1:
            p = p[:, 1]  # elasticidad para clase 1
    except Exception:
        # No pudo predecir, inicializar ceros
        p = np.zeros(len(df))

    p_mean = np.mean(p)
    # Parámetros del modelo
    params = model.params
    # Si es serie con índice incluyendo const
    betas = params[features]
    # Valor medio de variables
    x_mean = df[features].mean()
    # Elasticidad logit: beta * x_mean * p_mean * (1-p_mean)
    elasticities = betas * x_mean * p_mean * (1 - p_mean)
    result = pd.DataFrame({
        'variable': features,
        'elasticidad': elasticities.values
    })
    return result
