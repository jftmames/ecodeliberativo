import os
from openai import OpenAI

class DeliberationEngine:
    """
    Motor que descompone un análisis en subpreguntas jerarquizadas
    usando la nueva interfaz openai>=1.0.0.
    """

    def __init__(self, model_name: str = "gpt-4"):
        # Inicializa cliente; usa OPENAI_API_KEY de entorno o secretos de Streamlit
        self.client = OpenAI()
        self.model = model_name

    def generate_subquestions(self, prompt: str, features: list[str] | None = None) -> list[str]:
        """
        Genera subpreguntas a partir de un prompt de análisis y opcionalmente lista de variables.
        """
        system_msg = (
            "Eres un motor de indagación que descompone un análisis en subpreguntas jerarquizadas. "
            "Produce entre 3 y 7 preguntas claras y concisas."
        )
        user_msg = f"Análisis: {prompt}"
        if features:
            user_msg += f"\nVariables del modelo: {', '.join(features)}"

        # Llamada usando nuevo cliente OpenAI
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.6,
            max_tokens=256,
        )

        text = response.choices[0].message.content
        # Parsear texto en lista de preguntas
        questions = []
        for line in text.split("\n"):
            q = line.strip().lstrip("0123456789.)- ")
            if q:
                questions.append(q)
        return questions
