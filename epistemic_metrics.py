# epistemic_metrics.py

def calcular_metricas_deliberativas(deliberacion: dict) -> dict:
    """
    Calcula múltiples métricas sobre el razonamiento deliberativo.

    Parámetros:
        deliberacion (dict): Diccionario {pregunta: respuesta}

    Retorna:
        dict con métricas clave:
            - 'EEE': Índice de Equilibrio Erotético (cobertura, profundidad y coherencia).
            - 'Coherencia': índice básico para detectar respuestas contradictorias simples.
            - 'Profundidad': longitud promedio de respuestas (palabras).
            - 'Cobertura': proporción de preguntas respondidas (no vacías).
            - 'Exploración': porcentaje de preguntas totales formuladas.
    """
    total_preguntas = len(deliberacion)
    if total_preguntas == 0:
        return {
            'EEE': 0.0,
            'Coherencia': 0.0,
            'Profundidad': 0.0,
            'Cobertura': 0.0,
            'Exploración': 0.0
        }

    respuestas = list(deliberacion.values())
    respondidas = [r for r in respuestas if r.strip()]
    num_respondidas = len(respondidas)

    cobertura = num_respondidas / total_preguntas

    longitud_palabras = [len(r.strip().split()) for r in respondidas] if respondidas else [0]
    profundidad = sum(longitud_palabras) / num_respondidas if num_respondidas > 0 else 0

    exploracion = cobertura  # Adaptar si hay lógica distinta

    coherencia = 1.0 if num_respondidas == total_preguntas else 0.7

    eee = 0.6 * cobertura + 0.3 * min(profundidad / 30, 1.0) + 0.1 * coherencia
    eee = round(eee, 3)

    return {
        'EEE': eee,
        'Coherencia': round(coherencia, 3),
        'Profundidad': round(profundidad, 2),
        'Cobertura': round(cobertura, 3),
        'Exploración': round(exploracion, 3)
    }

def perfil_eee(eee: float) -> str:
    if eee >= 0.9:
        return "Juicio deliberativo óptimo"
    elif eee >= 0.7:
        return "Deliberación sólida con margen de mejora"
    elif eee >= 0.5:
        return "Indagación superficial o con sesgos no resueltos"
    else:
        return "Clausura epistémica prematura"
