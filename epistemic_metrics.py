# epistemic_metrics.py

from typing import Dict


def compute_eee(tracker_json: Dict, max_steps: int = 10) -> Dict[str, float]:
    """
    Calcula el Índice de Equilibrio Erotético (EEE) basado en el registro de razonamiento.
    Dimensiones:
      D1: Profundidad estructural
      D2: Pluralidad semántica
      D3: Trazabilidad razonadora
      D4: Reversibilidad deliberativa
      D5: Robustez ante disenso

    Parámetros:
      - tracker_json: diccionario con clave 'steps' que contiene una lista de pasos,
        cada uno con 'question', 'answer' y opcional 'metadata'.
      - max_steps: número máximo de pasos esperado para normalizar algunas métricas.

    Devuelve:
      Un diccionario con los valores de D1–D5 y la puntuación agregada 'EEE'.
    """
    steps = tracker_json.get("steps", [])
    n_steps = len(steps)

    # D1: Profundidad estructural
    # Mide cuántos pasos de indagación se han realizado, normalizado por max_steps.
    # Valores cercanos a 1 indican investigaciones profundas; valores bajos, indagación limitada.
    D1 = min(1.0, n_steps / max_steps) if max_steps > 0 else 0.0

    # D2: Pluralidad semántica
    # Refleja la diversidad de perspectivas exploradas.
    # Aquí usamos el mismo valor que D1 como placeholder;
    # en implementación avanzada, se podría contar distintas etiquetas de marco teórico.
    D2 = D1

    # D3: Trazabilidad razonadora
    # Proporción de pasos en los que el usuario proporcionó una respuesta.
    # Valores altos indican que casi todos los nodos de pregunta tienen una respuesta asociada.
    answered = sum(1 for s in steps if s.get("answer"))
    D3 = (answered / n_steps) if n_steps > 0 else 0.0

    # D4: Reversibilidad deliberativa
    # Frecuencia de reformulaciones detectadas en el proceso.
    # Calculamos la proporción de pasos marcados como reformulación en metadata.
    reformulations = sum(1 for s in steps if s.get("metadata", {}).get("reformulation", False))
    D4 = min(1.0, reformulations / max_steps) if max_steps > 0 else 0.0

    # D5: Robustez ante disenso
    # Evalúa la capacidad del sistema para mantener abiertas múltiples líneas de cuestionamiento.
    # Aquí usamos un placeholder constante; en mejoras futuras, podría medir cuántas
    # objeciones fueron consistentes o se integraron.
    D5 = 1.0

    # EEE agregado: media simple de las cinco dimensiones.
    EEE = (D1 + D2 + D3 + D4 + D5) / 5.0

    return {"D1": D1, "D2": D2, "D3": D3, "D4": D4, "D5": D5, "EEE": EEE}
