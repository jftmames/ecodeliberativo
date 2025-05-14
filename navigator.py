# navigator.py

from typing import Dict, Any, List
import streamlit as st

class EpistemicNavigator:
    """
    Registro y manejo del tracker epistémico de preguntas y respuestas.
    Usa el motor almacenado en st.session_state.engine.
    """

    @staticmethod
    def record(question: str, answer: str) -> None:
        """
        Asocia la respuesta a la última subpregunta sin respuesta que coincida.
        """
        engine = st.session_state.get("engine", None)
        if engine is None:
            # Nada que registrar si no existe motor
            return

        tracker = engine.get_tracker()
        # Recorremos de atrás hacia adelante para encontrar el paso pendiente
        for step in reversed(tracker["steps"]):
            if step["question"] == question and not step["answer"]:
                step["answer"] = answer
                return

    @staticmethod
    def get_tracker() -> Dict[str, List[Dict[str, Any]]]:
        """
        Devuelve el tracker completo (lista de pasos con pregunta, respuesta y metadata).
        """
        engine = st.session_state.get("engine", None)
        if engine is None:
            return {"steps": []}
        return engine.get_tracker()
