# mnl.py

import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

def fit_mnl(df: pd.DataFrame, features: list[str]) -> MNLogit:
    """
    Ajusta un modelo multinomial logit (MNL) a los datos.
    
    Parámetros:
    - df: DataFrame que contiene la variable dependiente 'Y' (categórica)
          y las explicativas listadas en `features`.
    - features: lista de nombres de columnas de df a usar como regresoras.
    
    Retorna:
    - model: objeto MNLogitResults ya ajustado (fit).
    """
    # Construir matriz de diseño con constante
    X = sm.add_constant(df[features], has_constant='add')
    y = df["Y"]
    
    # Ajustar el modelo
    mnl = MNLogit(y, X)
    result = mnl.fit(disp=False)
    return result

def predict_mnl(model: MNLogit, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Dada una instancia de MNLogitResults y un DataFrame de nuevas observaciones,
    calcula las probabilidades predichas para cada categoría de Y.
    
    Parámetros:
    - model: objeto MNLogitResults de statsmodels, previamente ajustado.
    - df: DataFrame con las mismas columnas en `features` para predecir.
    - features: lista de nombres de columnas en df usadas por el modelo.
    
    Retorna:
    - probs_df: DataFrame de forma (n_obs × n_categories) con las probabilidades
                predichas para cada categoría de la variable dependiente.
                Las columnas están nombradas según las categorías originales de Y.
    """
    # Reconstruir matriz de diseño para predicción
    Xnew = sm.add_constant(df[features], has_constant='add')
    
    # model.predict devuelve un numpy array de shape (n_obs, n_choices)
    probs = model.predict(Xnew)
    
    # Determinar las categorías originales de Y
    # model.model.endog contiene la serie de Y de ajuste
    categories = pd.Categorical(model.model.endog).categories
    
    # Construir DataFrame con columnas etiquetadas
    probs_df = pd.DataFrame(probs, columns=[str(cat) for cat in categories], index=df.index)
    return probs_df
