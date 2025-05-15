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
        DataFrame original que contiene las variables features y target.
    model : objeto
        Objeto entrenado con método .predict(exog).
    features : List[str]
        Lista de nombres de columnas usadas como regresores.
    target : str, opcional
        Nombre de la columna objetivo. Si no se especifica, se toma la última columna de df.
    """

    # --- 0) Reset de índice y copia ---
    data = df.reset_index(drop=True).copy()
    n_rows = data.shape[0]

    # --- 1) Inferir target si no se pasa ---
    if target is None:
        target = data.columns[-1]

    # --- 2) Preparar X e y ---
    X = data[features]
    y = data[target]

    # --- 3) Reconstruir exógeno (constante + features) ---
    exog = None
    try:
        exog_names = model.model.exog_names
        exog = pd.DataFrame(index=data.index)
        for name in exog_names:
            if name.lower() in ('const', 'intercept'):
                exog[name] = 1.0
            else:
                exog[name] = data[name]
    except Exception:
        exog = None

    # --- 4) Obtener predicciones ---
    base = exog if exog is not None else X
    try:
        preds = model.predict(base)
    except Exception:
        preds = model.predict(base.values)

    # --- 5) Aplanar y corregir mismatch múltiple ---
    preds = np.asarray(preds).flatten()
    if preds.size != n_rows:
        if preds.size % n_rows == 0:
            # “Desapilamos” y tomamos la primera tanda
            blocks = preds.size // n_rows
            preds = preds.reshape(blocks, n_rows)[0]
        else:
            raise ValueError(
                f"Predicciones ({preds.size}) no coinciden con filas de datos ({n_rows})."
            )

    # --- 6) Asignar predicciones y residuos ---
    data['pred']  = preds
    data['resid'] = data[target] - data['pred']

    # --- 7) Breusch–Pagan (heterocedasticidad) ---
    bp_exog = exog if exog is not None else X
    bp_test = het_breuschpagan(data['resid'], bp_exog)
    bp_pvalue = bp_test[3]

    # --- 8) Jarque–Bera (normalidad) ---
    _, jb_pvalue = stats.jarque_bera(data['resid'])

    # --- 9) VIF (multicolinealidad) ---
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
