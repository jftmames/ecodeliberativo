from typing import Dict

def compute_eee(tracker_json: Dict, max_steps: int = 10) -> Dict[str, float]:
    """
    Calcula el Índice de Equilibrio Erotético (EEE) basado en el registro de razonamiento.

    Dimensiones:
      D1: Profundidad estructural (número de pasos / max_steps)
      D2: Pluralidad semántica (idem D1; placeholder)
      D3: Trazabilidad (proporción de respuestas no vacías)
      D4: Reversibilidad (proporción de reformulaciones señaladas)
      D5: Robustez (placeholder constante = 1)

    Retorna:
      {"D1": ..., "D2": ..., "D3": ..., "D4": ..., "D5": ..., "EEE": ...}
    """
    steps = tracker_json.get("steps", [])
    n = len(steps)

    # D1: Profundidad
    D1 = min(1.0, n / max_steps) if max_steps > 0 else 0.0

    # D2: Pluralidad semántica
    D2 = D1

    # D3: Trazabilidad
    answered = sum(1 for s in steps if s.get("answer"))
    D3 = (answered / n) if n > 0 else 0.0

    # D4: Reversibilidad
    reform = sum(1 for s in steps if s.get("metadata", {}).get("reformulation", False))
    D4 = min(1.0, reform / max_steps) if max_steps > 0 else 0.0

    # D5: Robustez
    D5 = 1.0

    EEE = (D1 + D2 + D3 + D4 + D5) / 5.0

    return {"D1": D1, "D2": D2, "D3": D3, "D4": D4, "D5": D5, "EEE": EEE}
