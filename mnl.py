import pandas as pd
import statsmodels.api as sm

def fit_mnl(df: pd.DataFrame, features: list[str]):
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


def predict_mnl(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Calcula las probabilidades predichas por un modelo MNLogit.

    model: objeto resultado de fit_mnl.
    df: DataFrame con las mismas features.
    features: lista de nombres de columnas independientes.
    """
    X = sm.add_constant(df[features])
    probs = model.predict(X)
    # El modelo MNLogit retorna un array de probabilidades para cada clase
    return pd.DataFrame(probs, columns=model.model.endog_names)


