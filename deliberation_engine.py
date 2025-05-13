# deliberation_engine.py
from typing import List, Dict

class InquiryEngine:
    """
    Genera subpreguntas a partir de un texto base.
    (Placeholder: implementar con OpenAI/LangChain más adelante)
    """
    def __init__(self):
        pass

    def generate_subquestions(self, prompt: str) -> List[str]:
        # TODO: sustituir placeholder por llamada a API
        return [
            "¿Cómo influye el precio en la elección?",
            "¿Qué papel juega el ingreso?",
            "¿Hay efectos por edad?"
        ]

class ReasoningTracker:
    """
    Registra el flujo de razonamiento: preguntas planteadas, respuestas,
    decisiones y reformulaciones.
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
