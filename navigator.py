# navigator.py

import streamlit as st

class EpistemicNavigator:
    """
    Gestiona el registro y recuperación del tracker epistémico
    usando st.session_state para persistencia por sesión.
    """
    @staticmethod
    def _ensure_tracker():
        if "epistemic_tracker" not in st.session_state:
            st.session_state["epistemic_tracker"] = {"steps": []}

    @staticmethod
    def record(question: str, answer: str, metadata: dict = None):
        """
        Registra una nueva interacción (pregunta + respuesta + metadata opcional)
        en el tracker epistémico.
        """
        EpistemicNavigator._ensure_tracker()
        step = {
            "question": question,
            "answer": answer or "",
        }
        if metadata:
            step["metadata"] = metadata
        st.session_state["epistemic_tracker"]["steps"].append(step)

    @staticmethod
    def get_tracker() -> dict:
        """
        Devuelve el estado completo del tracker epistémico:
        {'steps': [ {question, answer, metadata?}, ... ] }
        """
        EpistemicNavigator._ensure_tracker()
        return st.session_state["epistemic_tracker"]

    @staticmethod
    def reset_tracker():
        """
        Reinicia el tracker, eliminando todos los pasos registrados.
        """
        st.session_state["epistemic_tracker"] = {"steps": []}
