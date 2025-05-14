from typing import List

class DeliberationEngine:
    """
    Motor de generación de subpreguntas para guiar la deliberación.
    Dado un prompt y un conjunto de variables, genera un conjunto de preguntas
    que estructuran el análisis epistémico.
    """
    def __init__(self):
        # En un futuro podría inicializar un modelo LLM u otros recursos.
        pass

    def generate_subquestions(self, prompt: str, features: List[str]) -> List[str]:
        """
        Genera subpreguntas a partir de un prompt y la lista de variables.

        :param prompt: descripción del análisis o pregunta raíz.
        :param features: lista de nombres de variables explicativas.
        :return: lista de subpreguntas orientadas a diferentes facetas del análisis.
        """
        subquestions: List[str] = []

        # 1. Generar preguntas sobre cada variable
        for feat in features:
            subquestions.append(
                f"¿Cómo influye '{feat}' en el resultado de la pregunta: '{prompt}'?"
            )

        # 2. Preguntas genéricas de profundización
        subquestions.append(
            f"¿Qué supuestos implícitos subyacen en la pregunta: '{prompt}'?"
        )
        subquestions.append(
            "¿Qué datos adicionales o fuentes de información serían necesarios para validar este análisis?"
        )
        subquestions.append(
            "¿Existen posibles contradicciones o escenarios alternativos que debamos considerar?"
        )

        # 3. Pregunta de revisión del enfoque
        subquestions.append(
            f"¿Deberíamos reformular la pregunta raíz ('{prompt}') para obtener un enfoque más claro?"
        )

        return subquestions
