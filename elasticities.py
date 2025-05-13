# elasticities.py

import numpy as np

def compute_logit_elasticities(model, df, features):
    """
    Calcula elasticidades puntuales promedio para un modelo Logit:
      E_j = β_j * x̄_j * (1 - p̄)
    """
    # Probabilidades medias
    p = model.predict()
    p_mean = np.mean(p)
    elasticities = {}
    for feature in features:
        beta = model.params.get(feature, 0.0)
        x_mean = df[feature].mean()
        elasticities[feature] = beta * x_mean * (1 - p_mean)
    return elasticities

def compute_loglog_elasticities(model, features):
    """
    Para un modelo OLS log-log, los coeficientes son elasticidades directas.
    """
    elasticities = {}
    for feature in features:
        elasticities[feature] = model.params.get(feature, 0.0)
    return elasticities

def compute_elasticities(model, df, features, model_type):
    """
    Wrapper: elige fórmula según modelo.
    """
    if model_type == 'Logit':
        return compute_logit_elasticities(model, df, features)
    elif model_type == 'LogLog':
        return compute_loglog_elasticities(model, features)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")

