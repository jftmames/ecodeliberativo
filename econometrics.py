# econometrics.py

import statsmodels.api as sm
import numpy as np

def estimate_ols(X, y):
    X = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X).fit()

def estimate_logit(X, y):
    X = sm.add_constant(X, has_constant="add")
    return sm.Logit(y, X).fit(disp=False)

def estimate_probit(X, y):
    X = sm.add_constant(X, has_constant="add")
    return sm.Probit(y, X).fit(disp=False)

def estimate_poisson(X, y):
    X = sm.add_constant(X, has_constant="add")
    return sm.Poisson(y, X).fit()

def estimate_tobit(X, y, left=0, right=np.inf):
    # Usa linearmodels para Tobit (debes instalar: pip install linearmodels)
    try:
        from linearmodels.iv import Tobit
    except ImportError:
        raise ImportError("Instala linearmodels para usar Tobit: pip install linearmodels")
    X = sm.add_constant(X, has_constant="add")
    mod = Tobit(y, X, left=left, right=right)
    res = mod.fit()
    return res

def estimate_nested_logit_pylogit(df_long, nests, alt_id_col="alt_id", obs_id_col="obs_id", choice_col="choice"):
    """
    Ajusta un modelo Nested Logit con pylogit sobre un DataFrame apilado.
    - df_long: DataFrame en formato long (apilado)
    - nests: dict {nest_id: [alt_id1, alt_id2, ...], ...}
    """
    try:
        import pylogit
    except ImportError:
        raise ImportError("Instala pylogit para usar Nested Logit: pip install pylogit")

    # Sólo se utiliza la variable 'precio' como ejemplo (ajusta aquí según tus features)
    specification = {col: ["all_alternatives"] for col in df_long.columns if col not in [alt_id_col, obs_id_col, choice_col, "nest"]}
    model = pylogit.create_choice_model(
        data=df_long,
        alt_id_col=alt_id_col,
        obs_id_col=obs_id_col,
        choice_col=choice_col,
        specification=specification,
        model_type="Nested Logit",
        names={col: col.capitalize() for col in specification.keys()},
        nest_spec=nests
    )
    model.fit_mle(init_vals=None)
    return model

def estimate_model(model_name, X, y, **kwargs):
    if model_name == "OLS":
        return estimate_ols(X, y)
    elif model_name == "Logit":
        return estimate_logit(X, y)
    elif model_name == "Probit":
        return estimate_probit(X, y)
    elif model_name == "Poisson":
        return estimate_poisson(X, y)
    elif model_name == "Tobit":
        return estimate_tobit(X, y, **kwargs)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
