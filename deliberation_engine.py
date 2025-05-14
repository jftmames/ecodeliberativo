# deliberation_engine.py

import hashlib
from typing import List, Dict

class DeliberationEngine:
    """
    Motor de generación de subpreguntas deliberativas.
    Mantiene un tracker interno con cada paso:
      - question: la subpregunta generada
      - answer: inicialmente None, se completa vía Navigator
      - metadata: puede incluir 'reformulation'=True si se genera sobre un prompt ya usado
    """

    def __init__(self):
        # Estructura: {'steps': [ {'question': str, 'answer': None, 'metadata': {...}}, ... ]}
        self.tracker: Dict[str, List[Dict]] = {"steps": []}
        # Guardamos el hash del último prompt para detectar reformulaciones
        self._last_prompt_hash: str = ""

    def generate_subquestions(self, prompt: str, features: List[str]) -> List[str]:
        """
        Dado un prompt y la lista de variables, genera una subpregunta por variable.
        Marca como 'reformulation' si el prompt cambia.
        """
        # Detectar reformulación
        h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        is_reformulation = (h != self._last_prompt_hash and bool(self.tracker["steps"]))
        self._last_prompt_hash = h

        subqs: List[str] = []
        # Generación simple: preguntar el efecto de cada característica
        for feat in features:
            q = f"¿Cómo influye '{feat}' en la probabilidad de consumo según tu modelo?"
            subqs.append(q)

        # Registrar cada paso en el tracker
        for q in subqs:
            self.tracker["steps"].append({
                "question": q,
                "answer": None,
                "metadata": {"reformulation": is_reformulation}
            })

        return subqs

    def get_tracker(self) -> Dict[str, List[Dict]]:
        """
        Devuelve el registro completo de subpreguntas y respuestas.
        """
        return self.tracker
