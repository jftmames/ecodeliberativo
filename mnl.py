import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

def fit_mnl(df: pd.DataFrame, features: list[str]):
    """
    Ajusta un modelo logit multinomial (MNLogit) a los datos.
    - df: DataFrame con las variables y la columna 'Y' de elección.
    - features: lista de nombres de columnas explicativas.
    """
    X = sm.add_constant(df[features], has_constant='add')
    y = df["Y"]
    model = MNLogit(y, X).fit(disp=False)
    return model

def predict_mnl(model, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Predice probabilidades con el modelo MNLogit.
    Devuelve un DataFrame donde cada columna es una categoría de Y.
    """
    X = sm.add_constant(df[features], has_constant='add')
    probs = model.predict(X)
    probs_df = pd.DataFrame(probs, columns=[f"P(Y={cat})" for cat in model.model.endog.unique()])
    return probs_df
