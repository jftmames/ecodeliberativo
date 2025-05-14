# navigator.py

import streamlit as st

class EpistemicNavigator:
    """
    Almacena y recupera el tracker de pasos de deliberación (pregunta/respuesta).
    """

    @staticmethod
    def record(question: str, answer: str, metadata: dict = None):
        """
        Registra un paso en st.session_state bajo 'navigator_steps'.
        """
        if metadata is None:
            metadata = {}
        step = {"question": question, "answer": answer, "metadata": metadata}
        if "navigator_steps" not in st.session_state:
            st.session_state.navigator_steps = []
        st.session_state.navigator_steps.append(step)

    @staticmethod
    def get_tracker() -> dict:
        """
        Devuelve el tracker con la lista de pasos.
        """
        return {"steps": st.session_state.get("navigator_steps", [])}
