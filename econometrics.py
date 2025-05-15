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

def estimate_nested_logit(X, y, nests, choices_col="choice"):
    """
    Nested logit con pylogit (instala con pip install pylogit).
    X: DataFrame con variables independientes.
    y: Serie o array de alternativas elegidas (enteros).
    nests: dict tipo {nombre_nest: [opciones_alternativa_en_nest, ...]}
    choices_col: nombre de la columna de alternativas elegidas.
    """
    try:
        import pylogit
    except ImportError:
        raise ImportError("Instala pylogit para usar Nested Logit: pip install pylogit")
    
    # pylogit espera un DataFrame especial con columnas 'obs_id_col', 'alt_id_col', 'choice_col'
    # Suponemos que cada fila es una alternativa para una observación
    # Si tienes datos apilados por individuo, adapta este bloque según tu estructura
    raise NotImplementedError("Adapta este bloque a tu formato de datos stacked (pylogit).")
    # Ejemplo (ajustar a tus datos):
    # model = pylogit.create_choice_model(
    #     data=df,
    #     alt_id_col='alt_id',
    #     obs_id_col='obs_id',
    #     choice_col=choices_col,
    #     specification=...,
    #     model_type="Nested Logit",
    #     names=...
    #     nest_spec=nests,
    # )
    # model.fit_mle()
    # return model

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
    elif model_name == "Nested Logit":
        return estimate_nested_logit(X, y, **kwargs)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")
