# deliberation_engine.py

from openai import OpenAI
from typing import List, Dict
import streamlit as st

class InquiryEngine:
    """
    Genera subpreguntas usando OpenAI a partir de un prompt y variables.
    Se basa en la nueva API client = OpenAI().
    """
    def __init__(self):
        # Inicializa el cliente con la clave de Streamlit Secrets
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        self.default_features = ["precio", "ingreso", "edad"]

    def generate_subquestions(self, prompt: str, features: List[str] = None) -> List[str]:
        if features is None:
            features = self.default_features

        system_msg = (
            "Eres un asistente econométrico que descompone un análisis "
            "de elección del consumidor en subpreguntas claras y jerarquizadas."
        )
        user_msg = (
            f"Análisis: {prompt}\n"
            f"Variables: {', '.join(features)}.\n"
            "Devuélveme 5 subpreguntas numeradas, sin explicaciones adicionales."
        )

        resp = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg}
            ],
            temperature=0.3,
            max_tokens=200
        )

        text = resp.choices[0].message.content.strip()
        subs = []
        for line in text.splitlines():
            line = line.strip()
            # extrae líneas que empiecen con dígito + punto
            if line and line[0].isdigit() and "." in line:
                subs.append(line.split(".", 1)[1].strip())
        return subs

class ReasoningTracker:
    """
    Registra el flujo de razonamiento: preguntas planteadas y sus respuestas.
    """
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
