# navigator.py

import streamlit as st
from typing import Dict

def create_navigation_dot(tracker_json: Dict) -> str:
    """
    Genera un string DOT para visualizar el flujo de razonamiento.
    Cada paso es representado como:
      - Q{i}: la pregunta (caja)
      - A{i}: la respuesta (óvalo)
    Conectando Q{i} -> A{i} y A{i} -> Q{i+1}.
    """
    dot = "digraph G {"
    steps = tracker_json.get("steps", [])
    for i, step in enumerate(steps):
        # Escapar comillas
        q_label = step["question"].replace('"','\\"')
        a_label = (step["answer"] or "Sin respuesta").replace('"','\\"')
        dot += f' Q{i} [label="{q_label}", shape=box];'
        dot += f' A{i} [label="{a_label}", shape=ellipse];'
        dot += f' Q{i} -> A{i};'
        if i < len(steps) - 1:
            dot += f' A{i} -> Q{i+1};'
    dot += "}"
    return dot

def display_navigation(tracker_json: Dict):
    """
    Muestra en Streamlit el diagrama de navegación de preguntas y respuestas.
    """
    dot = create_navigation_dot(tracker_json)
    st.graphviz_chart(dot)
