# econometrics.py

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit, MNLogit, Poisson

# Modelos econométricos para el análisis del comportamiento del consumidor

def estimate_ols(X, y):
    """
    Estima un modelo de regresión OLS.
    """
    X_ = sm.add_constant(X)
    model = sm.OLS(y, X_)
    results = model.fit()
    return results

def estimate_logit(X, y):
    """
    Estima un modelo Logit binario.
    """
    X_ = sm.add_constant(X)
    model = Logit(y, X_)
    results = model.fit()
    return results

def estimate_probit(X, y):
    """
    Estima un modelo Probit binario.
    """
    X_ = sm.add_constant(X)
    model = Probit(y, X_)
    results = model.fit()
    return results

def estimate_mnl(X, y):
    """
    Estima un modelo Logit Multinomial (MNL).
    """
    X_ = sm.add_constant(X)
    model = MNLogit(y, X_)
    results = model.fit()
    return results

def estimate_poisson(X, y):
    """
    Estima un modelo de regresión Poisson (conteo).
    """
    X_ = sm.add_constant(X)
    model = Poisson(y, X_)
    results = model.fit()
    return results

def estimate_model(model_name, X, y, **kwargs):
    """
    Selecciona y estima el modelo especificado.
    model_name: str, nombre del modelo ("OLS", "Logit", "Probit", "MNL", "Poisson")
    X: DataFrame de variables explicativas
    y: Serie/array objetivo
    kwargs: argumentos extra (por compatibilidad futura)
    """
    if model_name == "OLS":
        return estimate_ols(X, y)
    elif model_name == "Logit":
        return estimate_logit(X, y)
    elif model_name == "Probit":
        return estimate_probit(X, y)
    elif model_name == "MNL":
        return estimate_mnl(X, y)
    elif model_name == "Poisson":
        return estimate_poisson(X, y)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
