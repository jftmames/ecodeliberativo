import streamlit as st

class EpistemicNavigator:
    """
    Clase encargada de registrar y gestionar las respuestas a subpreguntas
    generadas por el motor de deliberación.
    """

    @staticmethod
    def record(question: str, answer: str) -> None:
        """
        Guarda en st.session_state un diccionario con preguntas y respuestas.
        """
        if "deliberation_log" not in st.session_state:
            st.session_state.deliberation_log = []
        st.session_state.deliberation_log.append({
            "question": question,
            "answer": answer
        })

    @staticmethod
    def get_log() -> list[dict]:
        """
        Devuelve el log completo de preguntas y respuestas.
        """
        return st.session_state.get("deliberation_log", [])
