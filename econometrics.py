import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import Logit, Probit, MNLogit, Poisson
from statsmodels.regression.linear_model import OLS
from statsmodels.miscmodels.tmodel import Tobit
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Any

def prepare_X_y(df, y_var, x_vars):
    X = df[x_vars]
    X = sm.add_constant(X, has_constant='add')
    y = df[y_var]
    return X, y

def fit_logit(df: pd.DataFrame, y_var: str, x_vars: list) -> Dict[str, Any]:
    X, y = prepare_X_y(df, y_var, x_vars)
    model = Logit(y, X)
    result = model.fit(disp=0)
    questions = [
        "¿Se cumplen los supuestos de independencia entre observaciones?",
        "¿Cómo interpretas los coeficientes en términos de probabilidades?",
        "¿Hay posibles variables omitidas o endogeneidad?"
    ]
    diagnostics = model_diagnostics(result, model_name="Logit")
    return {
        "modelo": "Logit",
        "result": result,
        "summary": result.summary().as_text(),
        "coef": result.params,
        "questions": questions,
        "diagnostics": diagnostics,
        "predicted": result.predict(X)
    }

def fit_probit(df: pd.DataFrame, y_var: str, x_vars: list) -> Dict[str, Any]:
    X, y = prepare_X_y(df, y_var, x_vars)
    model = Probit(y, X)
    result = model.fit(disp=0)
    questions = [
        "¿Los errores del modelo tienen distribución normal?",
        "¿Cómo afecta la elección del modelo (Probit vs Logit) a la interpretación?",
        "¿Qué variables podrían estar influyendo y no están en el modelo?"
    ]
    diagnostics = model_diagnostics(result, model_name="Probit")
    return {
        "modelo": "Probit",
        "result": result,
        "summary": result.summary().as_text(),
        "coef": result.params,
        "questions": questions,
        "diagnostics": diagnostics,
        "predicted": result.predict(X)
    }

def fit_tobit(df: pd.DataFrame, y_var: str, x_vars: list, left=0, right=np.inf) -> Dict[str, Any]:
    X, y = prepare_X_y(df, y_var, x_vars)
    model = Tobit(y, X, left=left, right=right)
    result = model.fit(disp=0)
    questions = [
        "¿Por qué se utiliza un modelo Tobit y no OLS?",
        "¿Qué efectos tienen los límites de censura en la interpretación de los coeficientes?",
        "¿Se justifica la hipótesis de normalidad de los errores?"
    ]
    diagnostics = model_diagnostics(result, model_name="Tobit")
    return {
        "modelo": "Tobit",
        "result": result,
        "summary": result.summary().as_text(),
        "coef": result.params,
        "questions": questions,
        "diagnostics": diagnostics,
        "predicted": result.predict(X)
    }

def fit_mnl(df: pd.DataFrame, y_var: str, x_vars: list) -> Dict[str, Any]:
    # Codificar la variable dependiente
    y_enc = LabelEncoder().fit_transform(df[y_var])
    X = sm.add_constant(df[x_vars], has_constant='add')
    model = MNLogit(y_enc, X)
    result = model.fit(disp=0)
    questions = [
        "¿Son razonables las categorías elegidas para la variable dependiente?",
        "¿Qué significa la base de referencia en este modelo?",
        "¿Hay independencia irrelevante de alternativas?"
    ]
    diagnostics = model_diagnostics(result, model_name="MNL")
    return {
        "modelo": "MNL",
        "result": result,
        "summary": result.summary().as_text(),
        "coef": result.params,
        "questions": questions,
        "diagnostics": diagnostics,
        "predicted": result.predict(X)
    }

def fit_ols(df: pd.DataFrame, y_var: str, x_vars: list) -> Dict[str, Any]:
    X, y = prepare_X_y(df, y_var, x_vars)
    model = OLS(y, X)
    result = model.fit()
    questions = [
        "¿Se cumple la homocedasticidad de los residuos?",
        "¿La relación entre variables es realmente lineal?",
        "¿Existen posibles problemas de multicolinealidad?"
    ]
    diagnostics = model_diagnostics(result, model_name="OLS")
    return {
        "modelo": "OLS",
        "result": result,
        "summary": result.summary().as_text(),
        "coef": result.params,
        "questions": questions,
        "diagnostics": diagnostics,
        "predicted": result.predict(X)
    }

def fit_poisson(df: pd.DataFrame, y_var: str, x_vars: list) -> Dict[str, Any]:
    X, y = prepare_X_y(df, y_var, x_vars)
    model = Poisson(y, X)
    result = model.fit(disp=0)
    questions = [
        "¿La variable dependiente representa conteos no negativos?",
        "¿Hay sobredispersión? ¿Sería mejor un modelo negativo binomial?",
        "¿El modelo ajusta bien los valores observados altos?"
    ]
    diagnostics = model_diagnostics(result, model_name="Poisson")
    return {
        "modelo": "Poisson",
        "result": result,
        "summary": result.summary().as_text(),
        "coef": result.params,
        "questions": questions,
        "diagnostics": diagnostics,
        "predicted": result.predict(X)
    }

# ---- UTILIDADES Y DIAGNÓSTICOS ----

def model_diagnostics(result, model_name="Modelo"):
    """Devuelve diccionario de diagnósticos clave para el informe deliberativo."""
    diags = {}
    try:
        if hasattr(result, 'resid'):
            resid = result.resid
            diags['media_residuos'] = np.mean(resid)
            diags['varianza_residuos'] = np.var(resid)
        if hasattr(result, 'prsquared'):
            diags['pseudo_R2'] = result.prsquared
        if hasattr(result, 'rsquared'):
            diags['R2'] = result.rsquared
        if hasattr(result, 'llf'):
            diags['log_likelihood'] = result.llf
        if hasattr(result, 'llnull'):
            diags['log_likelihood_null'] = result.llnull
        if hasattr(result, 'aic'):
            diags['AIC'] = result.aic
        if hasattr(result, 'bic'):
            diags['BIC'] = result.bic
    except Exception as e:
        diags['error'] = f"Error en diagnóstico: {str(e)}"
    return diags

# ---- GESTOR GENERAL DE MODELOS ----

def run_model(df: pd.DataFrame, modelo: str, y_var: str, x_vars: list, **kwargs):
    """Lanza el modelo adecuado según string 'modelo'."""
    if modelo == "Logit":
        return fit_logit(df, y_var, x_vars)
    elif modelo == "Probit":
        return fit_probit(df, y_var, x_vars)
    elif modelo == "Tobit":
        return fit_tobit(df, y_var, x_vars, **kwargs)
    elif modelo == "MNL":
        return fit_mnl(df, y_var, x_vars)
    elif modelo == "OLS":
        return fit_ols(df, y_var, x_vars)
    elif modelo == "Poisson":
        return fit_poisson(df, y_var, x_vars)
    else:
        raise ValueError("Modelo no implementado: " + modelo)

# ---- OPCIONAL: UTILIDAD PARA EXPORTAR COEFICIENTES A CSV ----

def export_coef_to_csv(result_dict: Dict[str, Any], filepath: str):
    coefs = result_dict.get("coef")
    if coefs is not None:
        if isinstance(coefs, pd.Series):
            coefs.to_csv(filepath)
        elif isinstance(coefs, pd.DataFrame):
            coefs.to_csv(filepath)
        else:
            pd.DataFrame(coefs).to_csv(filepath)
    else:
        raise ValueError("No hay coeficientes para exportar.")

# ---- OPCIONAL: ELASTICIDADES PARA MNL ----

def calculate_elasticities_mnl(result, X, y_enc):
    # Aquí puedes calcular elasticidades propias/cruzadas para MNL
    # Ejemplo simple (debes personalizar según tu aplicación)
    return {}

