# validation.py

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

def check_multicollinearity(df: pd.DataFrame, features: list) -> pd.DataFrame:
    X = sm.add_constant(df[features])
    vif_data = []
    for i, feature in enumerate(X.columns):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({"feature": feature, "VIF": vif})
    return pd.DataFrame(vif_data)

def check_heteroscedasticity(model):
    """
    Aplica Breusch–Pagan si el modelo tiene residuales continuos.
    """
    try:
        resid = model.resid
        exog = model.model.exog
    except Exception:
        return {"error": "Test no disponible para este tipo de modelo"}
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(resid, exog)
    return {
        "lm_stat": lm_stat,
        "lm_pvalue": lm_pvalue,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue
    }

def check_model_diagnostics(df: pd.DataFrame, model, features: list) -> dict:
    diagnostics = {}
    diagnostics["VIF"] = check_multicollinearity(df, features).to_dict(orient="records")
    diagnostics["heteroscedasticity"] = check_heteroscedasticity(model)
    # R-squared / pseudo-R²
    if hasattr(model, 'rsquared'):
        diagnostics["r_squared"] = model.rsquared
    elif hasattr(model, 'prsquared'):
        diagnostics["pseudo_r_squared"] = model.prsquared
    else:
        diagnostics["r_squared"] = None
    return diagnostics

