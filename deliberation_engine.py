# deliberation_engine.py

import os
import openai
from typing import List, Dict

openai.api_key = os.getenv("OPENAI_API_KEY")

class InquiryEngine:
    def __init__(self):
        self.default_features = ["precio", "ingreso", "edad"]

    def generate_subquestions(self, prompt: str, features: List[str] = None) -> List[str]:
        if features is None:
            features = self.default_features

        # Montamos un prompt para GPT
        sys_msg = (
            "Eres un asistente que descompone un problema econométrico "
            "sobre elección del consumidor en subpreguntas claras y jerarquizadas."
        )
        user_msg = (
            f"Problema: {prompt}\n"
            f"Variables: {', '.join(features)}.\n"
            "Devuélveme una lista de subpreguntas numeradas."
        )

        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": user_msg}
            ],
            temperature=0.7,
            max_tokens=300
        )
        text = resp.choices[0].message.content.strip()

        # Parsear las líneas que empiecen con “1.”, “2.”, etc.
        subs = []
        for line in text.splitlines():
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                subs.append(line.split(".", 1)[1].strip())
        return subs

class ReasoningTracker:
    def __init__(self):
        self.log: List[Dict] = []

    def add_step(self, question: str, answer: str, metadata: Dict = None):
        self.log.append({
            "question": question,
            "answer": answer,
            "metadata": metadata or {}
        })

    def to_json(self) -> Dict:
        return {"steps": self.log}


