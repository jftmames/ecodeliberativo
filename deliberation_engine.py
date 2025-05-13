# deliberation_engine.py
from typing import List, Dict

class InquiryEngine:
    """
    Genera subpreguntas a partir de un texto base.
    """
    def __init__(self):
        # variables por defecto si no se pasan
        self.default_features = ["precio", "ingreso", "edad"]

    def generate_subquestions(self, prompt: str, features: List[str] = None) -> List[str]:
        """
        Crea un listado de subpreguntas:
          - Una pregunta raíz
          - Una por cada feature
        """
        if features is None:
            features = self.default_features

        subs = []
        subs.append("¿Cuáles son los factores principales que influyen en la elección del consumidor?")
        for f in features:
            subs.append(f"¿Cómo influye '{f}' en la probabilidad de elección?")
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

