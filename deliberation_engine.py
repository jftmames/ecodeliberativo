# deliberation_engine.py

class DeliberationEngine:
    """
    Motor deliberativo: genera subpreguntas a partir de un prompt y lista de features.
    """
    def generate_subquestions(self, prompt: str, features: list) -> list:
        # Ejemplo sencillo: una subpregunta por cada feature
        subqs = []
        for feat in features:
            subqs.append(f"¿Cómo influye la variable '{feat}' en la respuesta principal?")
        # Puedes hacer esto más avanzado, por ejemplo, incluyendo combinaciones, contrafactuales, etc.
        return subqs
