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
    Diagnósticos para modelo de regresión:
    - Heterocedasticidad (Breusch–Pagan)
    - Normalidad de residuos (Jarque–Bera)
    - Multicolinealidad (VIF)
    """

    # Reset de índice y copia
    data = df.reset_index(drop=True).copy()
    n_rows = data.shape[0]

    # Inferir target si no se pasa
    if target is None:
        target = data.columns[-1]

    # Preparar X e y
    X = data[features]
    y = data[target]

    # Reconstruir exógeno si el modelo tiene exog_names (para statsmodels)
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

    # Obtener predicciones usando la matriz apropiada
    base = exog if exog is not None else X
    try:
        preds = model.predict(base)
    except Exception:
        preds = model.predict(base.values)

    # Aplanar y corregir mismatch múltiple
    preds = np.asarray(preds).flatten()
    if preds.size != data.shape[0]:
        if preds.size % data.shape[0] == 0:
            blocks = preds.size // data.shape[0]
            preds = preds.reshape(blocks, data.shape[0])[0]
        else:
            raise ValueError(
                f"Predicciones ({preds.size}) no coinciden con filas de datos ({data.shape[0]})."
            )

    # Asignar predicciones y residuos
    data['pred'] = preds
    data['resid'] = data[target] - data['pred']

    # 1) Heterocedasticidad: Breusch–Pagan
    bp_exog = exog if exog is not None else X
    bp_test = het_breuschpagan(data['resid'], bp_exog)
    bp_pvalue = bp_test[3]

    # 2) Normalidad de residuos: Jarque–Bera
    _, jb_pvalue = stats.jarque_bera(data['resid'])

    # 3) Multicolinealidad: VIF (solo si hay al menos 2 variables)
    vif_dict: Dict[str, float] = {}
    if len(features) > 1:
        X_vif = X.copy()
        # Elimina la constante si existe
        for col in ['const', 'intercept']:
            if col in X_vif.columns:
                X_vif = X_vif.drop(columns=[col])
        X_np = X_vif.values
        for i, feat in enumerate(X_vif.columns):
            try:
                vif_dict[feat] = variance_inflation_factor(X_np, i)
            except Exception:
                vif_dict[feat] = float('nan')
    else:
        vif_dict = {"(no aplica)": float('nan')}

    return {
        'bp_pvalue': bp_pvalue,
        'jb_pvalue': jb_pvalue,
        'vif': vif_dict,
        'residuals': data['resid']
    }
