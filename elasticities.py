# elasticities.py

import numpy as np

def calculate_elasticity_ols(model, variable, base_values, new_value, tracker=None):
    """Elasticidad para OLS: (Δy/y)/(Δx/x)"""
    base_input = base_values.copy()
    contra_input = base_values.copy()
    contra_input[variable] = new_value

    y0 = model.predict([base_input])[0]
    y1 = model.predict([contra_input])[0]

    delta_x = new_value - base_values[variable]
    delta_y = y1 - y0

    pct_change_x = delta_x / base_values[variable] if base_values[variable] != 0 else 0
    pct_change_y = delta_y / y0 if y0 != 0 else 0
    elasticity = pct_change_y / pct_change_x if pct_change_x != 0 else np.nan

    if tracker:
        tracker.log_step(
            question=f"¿Qué pasa si {variable} cambia de {base_values[variable]} a {new_value} (OLS)?",
            answer=f"Elasticidad OLS: {elasticity:.3f}",
            step_type="elasticidad_OLS",
            context={"y0": y0, "y1": y1, "delta_x": delta_x, "delta_y": delta_y, "elasticity": elasticity}
        )

    return elasticity


def calculate_elasticity_logit(model, variable, base_values, new_value, tracker=None):
    """Elasticidad para Logit: efecto en probabilidad"""
    base_input = base_values.copy()
    contra_input = base_values.copy()
    contra_input[variable] = new_value

    p0 = model.predict_proba([base_input])[0][1]  # Probabilidad de Y=1
    p1 = model.predict_proba([contra_input])[0][1]

    delta_x = new_value - base_values[variable]
    delta_p = p1 - p0

    pct_change_x = delta_x / base_values[variable] if base_values[variable] != 0 else 0
    pct_change_p = delta_p / p0 if p0 != 0 else 0
    elasticity = pct_change_p / pct_change_x if pct_change_x != 0 else np.nan

    if tracker:
        tracker.log_step(
            question=f"¿Qué pasa si {variable} cambia de {base_values[variable]} a {new_value} (Logit)?",
            answer=f"Elasticidad Logit: {elasticity:.3f}",
            step_type="elasticidad_Logit",
            context={"p0": p0, "p1": p1, "delta_x": delta_x, "delta_p": delta_p, "elasticity": elasticity}
        )

    return elasticity


def calculate_elasticity_probit(model, variable, base_values, new_value, tracker=None):
    """Elasticidad para Probit: cambio en probabilidad"""
    base_input = base_values.copy()
    contra_input = base_values.copy()
    contra_input[variable] = new_value

    p0 = model.predict_proba([base_input])[0][1]
    p1 = model.predict_proba([contra_input])[0][1]

    delta_x = new_value - base_values[variable]
    delta_p = p1 - p0

    pct_change_x = delta_x / base_values[variable] if base_values[variable] != 0 else 0
    pct_change_p = delta_p / p0 if p0 != 0 else 0
    elasticity = pct_change_p / pct_change_x if pct_change_x != 0 else np.nan

    if tracker:
        tracker.log_step(
            question=f"¿Qué pasa si {variable} cambia de {base_values[variable]} a {new_value} (Probit)?",
            answer=f"Elasticidad Probit: {elasticity:.3f}",
            step_type="elasticidad_Probit",
            context={"p0": p0, "p1": p1, "delta_x": delta_x, "delta_p": delta_p, "elasticity": elasticity}
        )

    return elasticity


def calculate_elasticity_mnl(model, variable, base_values, new_value, tracker=None):
    """
    Elasticidad para Multinomial Logit:
    Cambia la probabilidad de cada clase cuando una variable cambia
    """
    base_input = base_values.copy()
    contra_input = base_values.copy()
    contra_input[variable] = new_value

    p0 = model.predict_proba([base_input])[0]  # Array de probs por clase
    p1 = model.predict_proba([contra_input])[0]

    delta_x = new_value - base_values[variable]
    pct_change_x = delta_x / base_values[variable] if base_values[variable] != 0 else 0

    # Calcula elasticidad para cada clase
    elasticities = []
    for k, (prob0, prob1) in enumerate(zip(p0, p1)):
        pct_change_p = (prob1 - prob0) / prob0 if prob0 != 0 else 0
        elasticity = pct_change_p / pct_change_x if pct_change_x != 0 else np.nan
        elasticities.append(elasticity)

    if tracker:
        tracker.log_step(
            question=f"¿Qué pasa si {variable} cambia de {base_values[variable]} a {new_value} (MNL)?",
            answer=f"Elasticidad por clase: {['{:.3f}'.format(e) for e in elasticities]}",
            step_type="elasticidad_MNL",
            context={"p0": p0.tolist(), "p1": p1.tolist(), "elasticities": elasticities}
        )

    return elasticities  # Una elasticidad por clase


def calculate_elasticity_tobit(model, variable, base_values, new_value, tracker=None):
    """
    Elasticidad para Tobit (simplificada, dependiendo del paquete/modelo).
    Usualmente se calcula sobre el valor esperado truncado.
    """
    base_input = base_values.copy()
    contra_input = base_values.copy()
    contra_input[variable] = new_value

    y0 = model.predict([base_input])[0]
    y1 = model.predict([contra_input])[0]

    delta_x = new_value - base_values[variable]
    delta_y = y1 - y0

    pct_change_x = delta_x / base_values[variable] if base_values[variable] != 0 else 0
    pct_change_y = delta_y / y0 if y0 != 0 else 0
    elasticity = pct_change_y / pct_change_x if pct_change_x != 0 else np.nan

    if tracker:
        tracker.log_step(
            question=f"¿Qué pasa si {variable} cambia de {base_values[variable]} a {new_value} (Tobit)?",
            answer=f"Elasticidad Tobit: {elasticity:.3f}",
            step_type="elasticidad_Tobit",
            context={"y0": y0, "y1": y1, "elasticity": elasticity}
        )

    return elasticity


def calculate_elasticity_poisson(model, variable, base_values, new_value, tracker=None):
    """
    Elasticidad para modelos de conteo (Poisson):
    El coeficiente es la elasticidad directa (exp(coef) - 1) si log-link
    """
    base_input = base_values.copy()
    contra_input = base_values.copy()
    contra_input[variable] = new_value

    y0 = model.predict([base_input])[0]
    y1 = model.predict([contra_input])[0]

    delta_x = new_value - base_values[variable]
    delta_y = y1 - y0

    pct_change_x = delta_x / base_values[variable] if base_values[variable] != 0 else 0
    pct_change_y = delta_y / y0 if y0 != 0 else 0
    elasticity = pct_change_y / pct_change_x if pct_change_x != 0 else np.nan

    if tracker:
        tracker.log_step(
            question=f"¿Qué pasa si {variable} cambia de {base_values[variable]} a {new_value} (Poisson)?",
            answer=f"Elasticidad Poisson: {elasticity:.3f}",
            step_type="elasticidad_Poisson",
            context={"y0": y0, "y1": y1, "elasticity": elasticity}
        )

    return elasticity


def calculate_elasticity_nested_logit(model, variable, base_values, new_value, tracker=None):
    """
    Elasticidad para Nested Logit: generalización del MNL.
    Aquí se calcula el cambio en probabilidades de cada alternativa.
    """
    base_input = base_values.copy()
    contra_input = base_values.copy()
    contra_input[variable] = new_value

    p0 = model.predict_proba([base_input])[0]  # Array de probs por clase
    p1 = model.predict_proba([contra_input])[0]

    delta_x = new_value - base_values[variable]
    pct_change_x = delta_x / base_values[variable] if base_values[variable] != 0 else 0

    elasticities = []
    for k, (prob0, prob1) in enumerate(zip(p0, p1)):
        pct_change_p = (prob1 - prob0) / prob0 if prob0 != 0 else 0
        elasticity = pct_change_p / pct_change_x if pct_change_x != 0 else np.nan
        elasticities.append(elasticity)

    if tracker:
        tracker.log_step(
            question=f"¿Qué pasa si {variable} cambia de {base_values[variable]} a {new_value} (Nested Logit)?",
            answer=f"Elasticidad por clase: {['{:.3f}'.format(e) for e in elasticities]}",
            step_type="elasticidad_NestedLogit",
            context={"p0": p0.tolist(), "p1": p1.tolist(), "elasticities": elasticities}
        )

    return elasticities

# ----------- INTERFAZ UNIFICADA -----------

def calculate_elasticity(model, model_type, variable, base_values, new_value, tracker=None):
    """
    Interfaz unificada para calcular elasticidad deliberativa según el modelo.
    model_type: uno de "OLS", "Logit", "Probit", "MNL", "Tobit", "Poisson", "NestedLogit"
    """
    if model_type.lower() == "ols":
        return calculate_elasticity_ols(model, variable, base_values, new_value, tracker)
    elif model_type.lower() == "logit":
        return calculate_elasticity_logit(model, variable, base_values, new_value, tracker)
    elif model_type.lower() == "probit":
        return calculate_elasticity_probit(model, variable, base_values, new_value, tracker)
    elif model_type.lower() == "mnl":
        return calculate_elasticity_mnl(model, variable, base_values, new_value, tracker)
    elif model_type.lower() == "tobit":
        return calculate_elasticity_tobit(model, variable, base_values, new_value, tracker)
    elif model_type.lower() == "poisson":
        return calculate_elasticity_poisson(model, variable, base_values, new_value, tracker)
    elif model_type.lower() == "nestedlogit":
        return calculate_elasticity_nested_logit(model, variable, base_values, new_value, tracker)
    else:
        if tracker:
            tracker.log_step(
                question=f"Modelo no soportado: {model_type}",
                answer="Error: tipo de modelo no reconocido en elasticities.py",
                step_type="error_modelo",
                context={"model_type": model_type}
            )
        raise NotImplementedError(f"Tipo de modelo '{model_type}' no soportado en elasticities.py.")

