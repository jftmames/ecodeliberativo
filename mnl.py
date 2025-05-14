import pandas as pd
import numpy as np
import statsmodels.api as sm


def fit_mnl(df: pd.DataFrame, features: list[str]) -> sm.MNLogit:
    """
    Ajusta un modelo Logit Multinomial (MNLogit) a los datos.

    df: DataFrame con las columnas de features y la columna 'Y' como variable dependiente categórica.
    features: lista de nombres de columnas independientes.
    """
    # Matriz de regresores con constante
    X = sm.add_constant(df[features])
    # Variable dependiente
    y = df['Y']
    # Ajuste de MNLogit
    model = sm.MNLogit(y, X).fit(disp=False)
    return model


def predict_mnl(model: sm.MNLogit, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Calcula las probabilidades predichas por un modelo MNLogit.

    model: objeto resultado de fit_mnl.
    df: DataFrame con las mismas features.
    features: lista de nombres de columnas independientes.
    """
    X = sm.add_constant(df[features])
    probs = model.predict(X)
    # Obtener etiquetas de cada clase según el endog original
    endog = model.model.endog
    categories = sorted(np.unique(endog))
    # Asegurar que coincide el número de columnas
    if probs.shape[1] != len(categories):
        # Asignar nombres genéricos
        col_names = [f"class_{i}" for i in range(probs.shape[1])]
    else:
        col_names = [str(c) for c in categories]
    return pd.DataFrame(probs, columns=col_names)
