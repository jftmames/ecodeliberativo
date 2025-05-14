# deliberation_engine.py

class DeliberationEngine:
    """
    Motor simple para generar subpreguntas a partir de un prompt y lista de features.
    """

    def __init__(self):
        pass

    def generate_subquestions(self, prompt: str, features: list[str]) -> list[str]:
        """
        Genera subpreguntas heurísticas basadas en cada feature.
        """
        subqs = []
        for feat in features:
            subqs.append(f"¿Cómo influye '{prompt}' en la variable '{feat}'?")
        return subqs
