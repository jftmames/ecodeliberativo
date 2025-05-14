import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

def check_model_diagnostics(df: pd.DataFrame, model, features: list[str]) -> dict:
    """
    Genera un diccionario con diagnósticos clave del modelo estimado.

    Para Logit:
      - llf: Log-likelihood del modelo
      - llnull: Log-likelihood del modelo nulo
      - pseudo_R2: R2 de McFadden
      - AIC, BIC: criterios de información
    Para OLS:
      - R2, R2 ajustado
      - F estadístico y su p-valor
      - Durbin-Watson: autocorrelación de residuos
    Para MNL (MNLogit):
      - llf: Log-likelihood del modelo
      - no_of_params: número de parámetros estimados
      - AIC, BIC

    """
    res = {}
    model_name = type(model).__name__
    # Logit diagnostics
    if hasattr(model, 'llf') and hasattr(model, 'llnull'):
        llf = model.llf
        llnull = model.llnull
        pseudo_r2 = 1 - (llf / llnull) if llnull != 0 else np.nan
        res.update({
            'Model': 'Logit',
            'llf': llf,
            'llnull': llnull,
            'pseudo_R2': pseudo_r2,
            'AIC': model.aic,
            'BIC': model.bic,
        })
    # OLS diagnostics
    elif hasattr(model, 'rsquared'):
        dw = durbin_watson(model.resid)
        res.update({
            'Model': 'OLS',
            'R2': model.rsquared,
            'R2_adj': model.rsquared_adj,
            'F_statistic': model.fvalue,
            'F_pvalue': model.f_pvalue,
            'Durbin_Watson': dw,
        })
    # MNLogit diagnostics
    elif hasattr(model, 'model') and hasattr(model, 'llf'):
        # identificado como discrete_model.MNLogitResults
        res.update({
            'Model': 'MNL',
            'llf': model.llf,
            'no_of_params': int(model.df_model) + int(model.k_constant),
            'AIC': model.aic,
            'BIC': model.bic,
        })
    else:
        res['error'] = f"Modelo no reconocido: {model_name}"
    return res
