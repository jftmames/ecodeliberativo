import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

def fit_mnl(df: pd.DataFrame, features: list[str], target: str = "Y") -> sm.discrete.MNLogit:
    """
    Ajusta un modelo MNL (Multinomial Logit) usando statsmodels.

    Parámetros:
        df: DataFrame con datos.
        features: Lista de variables explicativas.
        target: Nombre de la variable dependiente categórica.

    Retorna:
        result: Resultado ajustado de MNLogit.
    """
    # Matriz de diseño
    X = sm.add_constant(df[features], prepend=True)
    # Variable dependiente
    y = df[target]
    # Ajuste del modelo
    model = MNLogit(y, X)
    result = model.fit(disp=False)
    return result


def predict_mnl(result: sm.discrete.MNLogit, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Predice probabilidades con un modelo MNL ajustado.

    Parámetros:
        result: Objeto resultado de fit_mnl.
        df: DataFrame con datos nuevos.
        features: Lista de variables explicativas.

    Retorna:
        DataFrame de probabilidades por alternativa.
    """
    # Matriz de predicción
    Xnew = sm.add_constant(df[features], prepend=True)
    # Predecir
    probs = result.predict(Xnew)
    # Determinar nombres de categorías
    categories = result.model.endog.unique()
    # Asegurar orden reproducible
    try:
        categories = sorted(categories)
    except TypeError:
        categories = list(categories)
    # Crear DataFrame
    prob_df = pd.DataFrame(probs, columns=[str(cat) for cat in categories])
    return prob_df
