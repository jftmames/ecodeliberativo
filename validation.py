# validation.py

def check_model_diagnostics(df, model, features):
    """
    Recoge métricas clave de diagnóstico: R2, pseudo-R2, AIC, BIC, etc.
    """
    diagnostics = {}
    # Si es OLS
    if hasattr(model, 'rsquared'):
        diagnostics["R_squared"] = model.rsquared
        diagnostics["Adj_R_squared"] = model.rsquared_adj
    # Si es Logit o MNLogit
    if hasattr(model, 'prsquared'):
        diagnostics["Pseudo_R_squared"] = model.prsquared
    # Ambos comparten
    diagnostics["AIC"] = model.aic
    diagnostics["BIC"] = model.bic
    return diagnostics
