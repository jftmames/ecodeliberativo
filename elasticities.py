# elasticities.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import List, Union

def compute_elasticities(model: sm.DiscreteResults, 
                         df: pd.DataFrame, 
                         features: List[str]
                        ) -> pd.DataFrame:
    """
    Calcula elasticidades promedio para un modelo Logit.
    
    Para cada variable explicativa x_k:
      E_k = β_k * x̄_k * p̄ * (1 - p̄)
    donde:
      - β_k      : coeficiente de la variable k
      - x̄_k     : media de la variable k en df
      - p̄       : probabilidad promedio predicha (P(Y=1))
    
    Devuelve un DataFrame con columnas ["Variable", "Elasticidad"].
    """
    # Construir matriz de diseño
    X = sm.add_constant(df[features], has_constant='add')
    # Probabilidades predichas
    p = model.predict(X)
    p_mean = np.mean(p)
    
    rows = []
    params = model.params  # incluye 'const'
    for feat in features:
        beta = params.get(feat, np.nan)
        x_mean = df[feat].mean()
        # Elasticidad promedio
        elas = beta * x_mean * p_mean * (1 - p_mean)
        rows.append({"Variable": feat, "Elasticidad": elas})
    
    elas_df = pd.DataFrame(rows)
    # Ordenar de mayor a menor magnitud
    elas_df["Magnitud"] = elas_df["Elasticidad"].abs()
    elas_df = elas_df.sort_values("Magnitud", ascending=False).drop(columns="Magnitud")
    elas_df = elas_df.reset_index(drop=True)
    return elas_df
