# validation.py

import numpy as np
import pandas as pd
from typing import Any, Dict, List
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_model_diagnostics(
    df: pd.DataFrame,
    model: Any,
    features: List[str],
    target: str = None
) -> Dict[str, Any]:
    """
    Ajusta diagnósticos básicos para un modelo de regresión.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las variables features y target.
    model : objeto
        Objeto entrenado con método .predict(exog).
    features : List[str]
        Lista de nombres de columnas usadas como regresores.
    target : str, opcional
        Nombre de la columna objetivo. Si no se especifica, se toma la última columna de df.

    Devuelve
    -------
    dict
        {
            'bp_pvalue': float,         # p-valor test Breusch–Pagan
            'jb_pvalue': float,         # p-valor test Jarque–Bera
            'vif': Dict[str, float],    # VIF por cada feature
            'residuals': pd.Series      # residuos calculados
        }
    """

    # Inferir target si no se pasa
    if target is None:
        target = df.columns[-1]

    # Preparamos X y y (sin constante)
    X = df[features]
    y = df[target]

    # Reconstruir exógeno tal cual el modelo lo espera
    # (solo para statsmodels con atributo model.exog_names)
    try:
        exog_names = model.model.exog_names
        exog = pd.DataFrame(index=df.index)
        for name in exog_names:
            if name.lower() in ['const', 'intercept']:
                exog[name] = 1.0
            else:
                exog[name] = df[name]
        preds = model.predict(exog)
    except Exception:
        # Fallback: intentar con DataFrame de features
        try:
            preds = model.predict(X)
        except Exception:
            # Último recurso: array numpy
            preds = model.predict(X.values)

    # Aplanar y chequear longitud
    preds = np.asarray(preds).flatten()
    if preds.shape[0] != df.shape[0]:
        raise ValueError(
            f"Predicciones ({preds.shape[0]}) no coinciden con filas de datos ({df.shape[0]})."
        )

    # Construimos DataFrame de diagnóstico
    data = df.reset_index(drop=True).copy()
    data['pred']  = preds
    data['resid'] = data[target] - data['pred']

    # 1) Heterocedasticidad: Breusch–Pagan
    bp_test    = het_breuschpagan(data['resid'], exog if 'exog' in locals() else X)
    bp_pvalue  = bp_test[3]

    # 2) Normalidad de residuos: Jarque–Bera
    jb_stat, jb_pvalue = stats.jarque_bera(data['resid'])

    # 3) Multicolinealidad: VIF (solo sobre X original)
    vif_dict: Dict[str, float] = {}
    X_np = X.values
    for i, feat in enumerate(features):
        vif_dict[feat] = variance_inflation_factor(X_np, i)

    return {
        'bp_pvalue': bp_pvalue,
        'jb_pvalue': jb_pvalue,
        'vif':       vif_dict,
        'residuals': data['resid']
    }
