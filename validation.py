import pandas as pd

def check_model_diagnostics(df: pd.DataFrame, model, features: list[str]) -> dict:
    """
    Devuelve métricas básicas según tipo de modelo:
     - Logit: AIC, BIC, Pseudo R²
     - OLS: R², AIC, BIC
     - MNL: AIC, BIC
    """
    name = type(model).__name__.lower()
    diag = {}

    if "logit" in name:
        diag["AIC"] = model.aic
        diag["BIC"] = model.bic
        diag["Pseudo R2"] = model.prsquared
    elif "ols" in name:
        diag["R2"] = model.rsquared
        diag["AIC"] = model.aic
        diag["BIC"] = model.bic
    elif "mnlogit" in name or "multinomial" in name:
        diag["AIC"] = model.aic
        diag["BIC"] = model.bic
    else:
        diag["Mensaje"] = "Modelo no reconocido para diagnóstico automático."

    return diag
