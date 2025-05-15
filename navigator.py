# navigator.py

from typing import Dict, Any, List

class EpistemicNavigator:
    _tracker: Dict[str, Any] = {"steps": []}

    @classmethod
    def add_step(cls, question: str, answer: str = None, metadata: Dict = None, parent: int = None):
        step = {
            "question": question,
            "answer": answer,
            "metadata": metadata or {},
            "parent": parent  # índice del padre en steps, o None si es raíz
        }
        cls._tracker["steps"].append(step)

    @classmethod
    def record(cls, question: str, answer: str, metadata: Dict = None, parent: int = None):
        idx = next(
            (i for i, s in enumerate(cls._tracker["steps"])
             if s["question"] == question and s.get("parent") == parent), 
            None
        )
        if idx is not None:
            cls._tracker["steps"][idx]["answer"] = answer
            if metadata:
                cls._tracker["steps"][idx]["metadata"].update(metadata)
        else:
            cls.add_step(question, answer, metadata, parent)

    @classmethod
    def get_tracker(cls) -> Dict[str, Any]:
        return cls._tracker

    @classmethod
    def clear_tracker(cls):
        cls._tracker = {"steps": []}
