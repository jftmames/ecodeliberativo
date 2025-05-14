from typing import List

class DeliberationEngine:
    """
    Motor simple de generación de subpreguntas
    basado en la lista de características.
    """

    def __init__(self):
        pass

    def generate_subquestions(self, prompt: str, features: List[str]) -> List[str]:
        """
        A partir de un prompt y las variables del modelo,
        genera una subpregunta por cada variable:
        Ej: "Analiza cómo {feat} influye en {prompt}"
        """
        subqs = []
        for feat in features:
            subqs.append(f"¿Cómo afecta '{feat}' a {prompt.lower()}?")
        return subqs
