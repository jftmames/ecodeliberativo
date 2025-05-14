# deliberation_engine.py

from typing import List
from navigator import EpistemicNavigator

class DeliberationEngine:
    """
    Motor de indagación deliberativa.
    Genera subpreguntas a partir de un prompt y un conjunto de features,
    y registra cada paso en el EpistemicNavigator.
    """

    def __init__(self):
        # No mantenemos estado interno: todo va al navigator
        pass

    def generate_subquestions(self, prompt: str, features: List[str]) -> List[str]:
        """
        Dado un prompt libre y una lista de features (variables explicativas),
        construye preguntas específicas que ayuden a explorar el análisis.
        Ejemplo: para prompt="Analizar la probabilidad de compra"
        y features=["precio","ingreso"], genera:
          - "¿Cómo influye el precio en la probabilidad de compra?"
          - "¿Qué efecto tiene el ingreso en la probabilidad de compra?"
        """
        subqs = []
        base = prompt.strip().rstrip(".?¡!")
        for feat in features:
            question = f"¿Cómo influye **{feat}** en {base}?"
            subqs.append(question)
        # Registrar las preguntas sin respuesta aún
        for q in subqs:
            EpistemicNavigator.add_step(question=q, answer=None, metadata={})
        return subqs
