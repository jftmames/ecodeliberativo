# deliberation_engine.py
import os
import openai

class DeliberationEngine:
    """
    Motor que, dada una descripción del análisis, genera
    automáticamente subpreguntas jerarquizadas usando la API de OpenAI.
    """

    def __init__(self, model_name: str = "gpt-4"):
        # Se asume que has configurado OPENAI_API_KEY en las secrets de Streamlit
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model_name

    def generate_subquestions(self, prompt: str, features: list[str] | None = None) -> list[str]:
        """
        Genera una lista de subpreguntas basadas en el prompt y, opcionalmente, las variables del modelo.
        """
        # Contexto de sistema: orientar al LLM a producir subpreguntas
        system_msg = (
            "Eres un motor de indagación que descompone un análisis en subpreguntas jerarquizadas. "
            "Produces entre 3 y 7 preguntas claras y concisas."
        )
        # Construimos el mensaje de usuario incluyendo las features si las hay
        user_msg = f"Análisis: {prompt}"
        if features:
            user_msg += f"\nVariables del modelo: {', '.join(features)}"

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.6,
            max_tokens=256
        )

        text = response.choices[0].message.content
        # Parseamos líneas en lista de preguntas (quitando guiones o numeración)
        lines = []
        for raw in text.split("\n"):
            q = raw.strip()
            if not q:
                continue
            # Eliminar prefijos tipo "1. " o "- "
            if q[0].isdigit() and q[1] in ".)":
                q = q[2:].strip()
            if q.startswith("-"):
                q = q[1:].strip()
            lines.append(q)
        return lines
