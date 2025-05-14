import streamlit as st
from typing import Dict, List

class EpistemicNavigator:
    """
    Registra y recupera el historial de preguntas y respuestas
    para luego calcular métricas epistémicas.
    """

    @staticmethod
    def _ensure_tracker():
        if "navigator_tracker" not in st.session_state:
            st.session_state.navigator_tracker = {"steps": []}

    @classmethod
    def record(cls, question: str, answer: str, metadata: Dict = None):
        """
        Agrega un paso al tracker:
          - question: texto de la subpregunta
          - answer: texto de la respuesta
          - metadata: diccionario opcional (ej. reformulation)
        """
        cls._ensure_tracker()
        step = {
            "question": question,
            "answer": answer or "",
            "metadata": metadata or {}
        }
        st.session_state.navigator_tracker["steps"].append(step)

    @classmethod
    def get_tracker(cls) -> Dict:
        """
        Devuelve el diccionario {'steps': [...]}
        """
        cls._ensure_tracker()
        return st.session_state.navigator_tracker
