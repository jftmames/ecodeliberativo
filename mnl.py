import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

def fit_mnl(df: pd.DataFrame, features: list[str]):
    """
    Ajusta un modelo Logit Multinomial a los datos.
    - df: DataFrame con variables explicativas y la variable Y categórica.
    - features: lista de nombres de columnas explicativas.
    """
    X = sm.add_constant(df[features])
    y = df["Y"]
    model = MNLogit(y, X).fit(disp=False)
    return model

def predict_mnl(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Devuelve un DataFrame con las probabilidades predichas para cada categoría de Y.
    Las columnas serán P(Y=valor) para cada valor único de Y en los datos del modelo.
    """
    X = sm.add_constant(df[features], has_constant='add')
    probs = model.predict(X)
    # etiquetas de columna según categorías de Y
    cats = model.model.endog.unique()
    col_names = [f"P(Y={cat})" for cat in cats]
    probs_df = pd.DataFrame(probs, columns=col_names, index=df.index)
    return probs_df
