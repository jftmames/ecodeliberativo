# mnl.py

import pandas as pd
import statsmodels.api as sm
import numpy as np

def fit_mnl(df: pd.DataFrame, features: list[str], endog_name: str = "Y"):
    """
    Ajusta un modelo de regresión logística multinomial (MNLogit).
    - df: DataFrame con las variables explicativas y la variable dependiente.
    - features: lista de nombres de columnas explicativas.
    - endog_name: nombre de la columna dependiente (categorías).
    """
    # Construye diseño con constante
    X = sm.add_constant(df[features], has_constant="add")
    y = df[endog_name]
    model = sm.MNLogit(y, X).fit(disp=False)
    return model

def predict_mnl(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Dada la tabla de datos y el modelo entrenado, devuelve un DataFrame
    n_obs × n_categorías con las probabilidades P(Y=cat).
    """
    # Reconstruye X con constante
    X = sm.add_constant(df[features], has_constant="add")
    # statsmodels MNLogit.predict regresa un array (n_obs, n_cats)
    probs = model.predict(X)  # numpy array

    # Extraemos las categorías originales del endog en el modelo
    # model.model.endog es el array de entrenamiento
    cats = np.unique(model.model.endog)
    cats.sort()

    # Nombramos columnas
    col_names = [f"P(Y={cat})" for cat in cats]

    # Construimos el DataFrame final
    probs_df = pd.DataFrame(probs, columns=col_names, index=df.index)

    return probs_df
