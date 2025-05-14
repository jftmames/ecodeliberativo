from typing import List

class DeliberationEngine:
    """
    Motor de generación de subpreguntas para guiar la investigación.
    Usa heurísticas simples sobre el prompt y las variables seleccionadas.
    """
    def generate_subquestions(self, prompt: str, features: List[str]) -> List[str]:
        subqs = []
        # preguntas sobre cada variable
        for feat in features:
            subqs.append(
                f"¿Cómo influye **{feat}** en la respuesta al prompt: '{prompt}'?"
            )
        # preguntas contextualizantes
        subqs.append(f"¿Qué supuestos subyacentes consideras al abordar: '{prompt}'?")
        subqs.append("¿Qué limitaciones metodológicas podrían afectar tu análisis?")
        return subqs
