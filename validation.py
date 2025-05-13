# validation.py

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

def check_multicollinearity(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Calcula el VIF (Variance Inflation Factor) para cada feature.
    Devuelve un DataFrame con columnas ['feature', 'VIF'].
    """
    X = sm.add_constant(df[features])
    vif_data = []
    for i, feature in enumerate(X.columns):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({"feature": feature, "VIF": vif})
    return pd.DataFrame(vif_data)

def check_heteroscedasticity(model) -> dict:
    """
    Aplica la prueba de Breusch-Pagan para heteroscedasticidad.
    Devuelve un dict con keys: ['lm_stat', 'lm_pvalue', 'f_stat', 'f_pvalue'].
    """
    resid = model.resid
    exog = model.model.exog
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(resid, exog)
    return {
        "lm_stat": lm_stat,
        "lm_pvalue": lm_pvalue,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue
    }

def check_model_diagnostics(df: pd.DataFrame, model, features: list) -> dict:
    """
    Realiza un conjunto de pruebas de diagnóstico econométrico:
      - Multicolinealidad (VIF)
      - Heteroscedasticidad (Breusch-Pagan)
      - R-squared (si aplica)
    Devuelve un dict con resultados resumidos.
    """
    diagnostics = {}
    diagnostics["VIF"] = check_multicollinearity(df, features).to_dict(orient="records")
    diagnostics["heteroscedasticity"] = check_heteroscedasticity(model)
    # R-squared solo para modelos OLS
    if hasattr(model, 'rsquared'):
        diagnostics["r_squared"] = model.rsquared
    else:
        diagnostics["pseudo_r_squared"] = model.prsquared  # para Logit
    return diagnostics
