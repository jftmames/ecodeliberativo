from typing import Dict

def compute_eee(tracker_json: Dict, max_steps: int = 10) -> Dict[str, float]:
    """
    Calcula el Índice de Equilibrio Erotético (EEE) basado en el registro de razonamiento.
    Dimensiones:
      D1: Profundidad estructural (número de pasos / max_steps)
      D2: Pluralidad semántica (igual a D1, placeholder)
      D3: Trazabilidad (proporción de respuestas no vacías)
      D4: Reversibilidad (frecuencia de reformulaciones)
      D5: Robustez (placeholder = 1)
    """
    steps = tracker_json.get("steps", [])
    n_steps = len(steps)

    # D1: profundidad normalizada
    D1 = min(1.0, n_steps / max_steps) if max_steps > 0 else 0.0

    # D2: pluralidad semántica (mismo valor que D1 aquí)
    D2 = D1

    # D3: trazabilidad = proporción de pasos con respuesta no vacía
    answered = sum(1 for s in steps if s.get("answer"))
    D3 = (answered / n_steps) if n_steps > 0 else 0.0

    # D4: reversibilidad = proporción de pasos marcados como reformulación
    reformulations = sum(1 for s in steps if s.get("metadata", {}).get("reformulation", False))
    D4 = min(1.0, reformulations / max_steps) if max_steps > 0 else 0.0

    # D5: robustez (se asume máxima como placeholder)
    D5 = 1.0

    # EEE agregado
    EEE = (D1 + D2 + D3 + D4 + D5) / 5.0

    return {
        "D1 Profundidad": D1,
        "D2 Pluralidad": D2,
        "D3 Trazabilidad": D3,
        "D4 Reversibilidad": D4,
        "D5 Robustez": D5,
        "EEE": EEE
    }
