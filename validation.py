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

    # 0) Reset de índice y copia
    data = df.reset_index(drop=True).copy()

    # 1) Inferir target si no se pasa
    if target is None:
        target = data.columns[-1]

    # 2) Preparar X e y sobre el DataFrame reinicializado
    X = data[features]
    y = data[target]

    # 3) Reconstruir la matriz exógena tal cual la usó el modelo (incluye constante)
    exog = None
    try:
        exog_names = model.model.exog_names
        exog = pd.DataFrame(index=data.index)
        for name in exog_names:
            if name.lower() in ('const', 'intercept'):
                exog[name] = 1.0
            else:
                # asumimos que la columna existe en data
                exog[name] = data[name]
    except Exception:
        # si falla, seguiremos sin exog explícito
        exog = None

    # 4) Obtener predicciones usando la matriz apropiada
    try:
        preds = model.predict(exog if exog is not None else X)
    except Exception:
        # en último recurso, pasar numpy array
        preds = model.predict((exog if exog is not None else X).values)

    # 5) Aplanar y comprobar longitud
    preds = np.asarray(preds).flatten()
    if preds.size != data.shape[0]:
        raise ValueError(
            f"Predicciones ({preds.size}) no coinciden con filas de datos ({data.shape[0]})."
        )

    # 6) Asignar predicciones y residuos
    data['pred']  = preds
    data['resid'] = data[target] - data['pred']

    # 7) Heterocedasticidad: Breusch–Pagan
    bp_exog = exog if exog is not None else X
    bp_test = het_breuschpagan(data['resid'], bp_exog)
    bp_pvalue = bp_test[3]

    # 8) Normalidad de residuos: Jarque–Bera
    _, jb_pvalue = stats.jarque_bera(data['resid'])

    # 9) Multicolinealidad: VIF (sobre X original)
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
