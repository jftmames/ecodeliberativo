# epistemic_metrics.py

from typing import Dict

def compute_eee(tracker_json: Dict, max_steps: int = 10) -> Dict[str, float]:
    """
    Calcula el Índice de Equilibrio Erotético (EEE) a partir del registro de razonamiento.

    Dimensiones evaluadas (0–1):
      D1. Profundidad estructural:
          Proporción de pasos realizados respecto a max_steps.
          Indica cuánto se ha explorado el árbol de indagación.
      D2. Pluralidad semántica:
          Igual a D1 aquí (placeholder), pero debería medir número de perspectivas distintas.
          Refleja cuántas visiones o marcos se han considerado.
      D3. Trazabilidad razonadora:
          Proporción de pasos con respuesta no vacía.
          Evalúa si cada pregunta tiene una respuesta documentada.
      D4. Reversibilidad deliberativa:
          Proporción de pasos marcados como reformulación.
          Mide cuántas veces se revisó o ajustó el foco interrogativo.
      D5. Robustez ante disenso:
          Constante =1 (placeholder), idealmente mediría resistencia de respuestas a objeciones.

    EEE agregado:
      Promedio simple de D1–D5, refleja la calidad estructural y dinámica de la deliberación.

    Parámetros:
      - tracker_json: {'steps': [ {'question','answer','metadata': {...}}, ... ]}
      - max_steps: número máximo de pasos esperados para normalizar D1 y D4.

    Retorna:
      Diccionario con valores de D1, D2, D3, D4, D5 y EEE agregado.
    """
    steps = tracker_json.get("steps", [])
    n_steps = len(steps)

    # D1: Profundidad estructural normalizada
    D1 = min(1.0, n_steps / max_steps) if max_steps > 0 else 0.0

    # D2: Pluralidad semántica (placeholder igual a D1)
    D2 = D1

    # D3: Trazabilidad = % de pasos que tienen respuesta
    answered = sum(1 for step in steps if step.get("answer"))
    D3 = (answered / n_steps) if n_steps > 0 else 0.0

    # D4: Reversibilidad = % de pasos marcados como reformulación
    reformulations = sum(1 for step in steps if step.get("metadata", {}).get("reformulation", False))
    D4 = min(1.0, reformulations / max_steps) if max_steps > 0 else 0.0

    # D5: Robustez ante disenso (placeholder constante)
    D5 = 1.0

    # EEE: promedio de todas las dimensiones
    EEE = (D1 + D2 + D3 + D4 + D5) / 5.0

    return {
        "D1 Profundidad": D1,
        "D2 Pluralidad": D2,
        "D3 Trazabilidad": D3,
        "D4 Reversibilidad": D4,
        "D5 Robustez": D5,
        "EEE": EEE
    }
