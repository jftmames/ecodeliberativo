import numpy as np
import pandas as pd

def check_model_diagnostics(df: pd.DataFrame, model, features: list[str]) -> dict:
    """
    Devuelve un diccionario con métricas de diagnóstico según el tipo de modelo:
      - Logit binario: pseudo R2
      - OLS: R2 y RMSE
      - MNL: pseudo R2
    """
    res = {}
    mod_name = type(model.model).__name__.lower()

    if "logit" in mod_name:
        # Pseudo R² de McFadden
        res["Pseudo R2"] = round(model.prsquared, 4)

    elif "ols" in mod_name:
        res["R2"] = round(model.rsquared, 4)
        resid = model.resid
        res["RMSE"] = round(np.sqrt(np.mean(resid**2)), 4)

    elif "mnlogit" in mod_name or "multinomial" in mod_name:
        res["Pseudo R2"] = round(model.prsquared, 4)

    # VIF para detectar multicolinealidad (solo para OLS/Logit)
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import statsmodels.api as sm

        X = sm.add_constant(df[features], has_constant='add')
        vif_data = {feat: variance_inflation_factor(X.values, i+1)
                    for i, feat in enumerate(features)}
        res["VIF"] = {feat: round(v, 2) for feat, v in vif_data.items()}
    except Exception:
        pass

    return res

