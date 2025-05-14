# deliberation_engine.py

import os
from openai import OpenAI  # nuevo cliente
from openai import api_requestor, api_resources

class DeliberationEngine:
    def __init__(self, model_name: str = "gpt-4"):
        self.client = OpenAI()  # lee automáticamente OPENAI_API_KEY
        self.model = model_name

    def generate_subquestions(self, prompt: str, features: list[str] | None = None) -> list[str]:
        system_msg = (
            "Eres un motor de indagación que descompone un análisis en subpreguntas jerarquizadas. "
            "Produce entre 3 y 7 preguntas claras y concisas."
        )
        user_msg = f"Análisis: {prompt}"
        if features:
            user_msg += f"\nVariables del modelo: {', '.join(features)}"

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.6,
            max_tokens=256
        )

        text = resp.choices[0].message.content
        # parsear líneas a lista
        questions = []
        for line in text.split("\n"):
            q = line.strip().lstrip("0123456789.)- ").strip()
            if q:
                questions.append(q)
        return questions
