# epistemic_metrics.py

def calcular_eee(deliberacion: dict) -> float:
    """
    Calcula un EEE básico: porcentaje de preguntas respondidas y longitud media de respuesta.
    Puedes evolucionarlo a una métrica más compleja según los criterios teóricos del Código Deliberativo.
    """
    total = len(deliberacion)
    if total == 0:
        return 0.0
    respondidas = sum(1 for r in deliberacion.values() if r.strip())
    profundidad = sum(len(r.strip().split()) for r in deliberacion.values() if r.strip())
    longitud_media = profundidad / respondidas if respondidas else 0
    # Peso: 70% preguntas respondidas, 30% profundidad (ajusta a tu gusto)
    eee = (0.7 * (respondidas/total) + 0.3 * min(longitud_media/30, 1.0))
    return round(eee, 2)

def perfil_eee(eee: float) -> str:
    if eee >= 0.9:
        return "Juicio deliberativo óptimo"
    elif eee >= 0.7:
        return "Deliberación sólida con margen de mejora"
    elif eee >= 0.5:
        return "Indagación superficial o con sesgos no resueltos"
    else:
        return "Clausura epistémica prematura"
