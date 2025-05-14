# navigator.py

from typing import Dict, List

class EpistemicNavigator:
    """
    Registra y expone el historial de pasos deliberativos:
      cada paso es un dict con 'question', 'answer' y 'metadata'.
    """

    _tracker: Dict[str, List[Dict]] = {"steps": []}

    @classmethod
    def add_step(cls, question: str, answer: str = None, metadata: Dict = None):
        """
        Añade un nuevo paso al tracker.
        - question: enunciado de la subpregunta.
        - answer: respuesta proporcionada (o None si aún no hay).
        - metadata: info adicional, p.ej. {'reformulation': True}.
        """
        cls._tracker["steps"].append({
            "question": question,
            "answer": answer,
            "metadata": metadata or {}
        })

    @classmethod
    def record(cls, question: str, answer: str, metadata: Dict = None):
        """
        Si la última pregunta coincide, actualiza su respuesta y metadata;
        si no, crea un nuevo paso.
        """
        if cls._tracker["steps"] and cls._tracker["steps"][-1]["question"] == question:
            cls._tracker["steps"][-1]["answer"] = answer
            cls._tracker["steps"][-1]["metadata"].update(metadata or {})
        else:
            cls.add_step(question, answer, metadata)

    @classmethod
    def get_tracker(cls) -> Dict[str, List[Dict]]:
        """
        Devuelve el diccionario completo {'steps': [...]}.
        """
        return cls._tracker

    @classmethod
    def clear_tracker(cls):
        """
        Limpia
