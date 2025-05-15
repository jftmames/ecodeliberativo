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
    target: str
) -> Dict[str, Any]:
    """
    Ajusta diagnósticos básicos para un modelo de regresión.
    
    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las variables features y target.
    model : objeto
        Objeto entrenado con método .predict(X).
    features : List[str]
        Lista de nombres de columnas usadas como regresores.
    target : str
        Nombre de la columna objetivo.
    
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

    # 1) Preparar X e y
    X = df[features]
    y = df[target]

    # 2) Calcular predicciones robustamente
    try:
        preds = model.predict(X)
    except Exception:
        # En caso de que model.predict espere array numpy
        preds = model.predict(X.values)

    # Forzar un array 1D y comprobar longitud
    preds = np.asarray(preds).flatten()
    if preds.shape[0] != df.shape[0]:
        raise ValueError(
            f"Cantidad de predicciones ({preds.shape[0]}) "
            f"no coincide con filas de datos ({df.shape[0]})."
        )

    # 3) Construir DataFrame de análisis
    data = df.reset_index(drop=True).copy()
    data['pred'] = preds
    data['resid'] = data[target] - data['pred']

    # 4) Heterocedasticidad: Breusch–Pagan
    #    Regresores en forma de matriz (statsmodels los requiere así)
    bp_test = het_breuschpagan(data['resid'], X)
    bp_pvalue = bp_test[3]

    # 5) Normalidad de residuos: Jarque–Bera
    jb_stat, jb_pvalue = stats.jarque_bera(data['resid'])

    # 6) Multicolinealidad: VIF
    vif_dict: Dict[str, float] = {}
    X_np = X.values
    for i, feat in enumerate(features):
        vif_dict[feat] = variance_inflation_factor(X_np, i)

    return {
        'bp_pvalue': bp_pvalue,
        'jb_pvalue': jb_pvalue,
        'vif': vif_dict,
        'residuals': data['resid']
    }
