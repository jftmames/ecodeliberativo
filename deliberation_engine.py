# deliberation_engine.py

import streamlit as st

def preguntar_deliberativo(preguntas, session_key="deliberation_answers"):
    """
    Muestra cada pregunta, permite al usuario responder y guarda las respuestas en session_state.
    Retorna diccionario {pregunta: respuesta}.
    """
    respuestas = st.session_state.get(session_key, {})
    nuevas_respuestas = {}
    st.markdown("### Responde las preguntas deliberativas:")
    for i, pregunta in enumerate(preguntas, 1):
        key = f"{session_key}_{i}"
        respuesta = st.text_area(f"{i}. {pregunta}", value=respuestas.get(pregunta, ""), key=key)
        nuevas_respuestas[pregunta] = respuesta
    st.session_state[session_key] = nuevas_respuestas
    return nuevas_respuestas
