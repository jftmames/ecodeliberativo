# visualizations.py

import matplotlib.pyplot as plt
import numpy as np

def plot_choice_probabilities(probs_df, show=False):
    """
    Dibuja las probabilidades de elección para cada alternativa.
    - probs_df: DataFrame donde cada columna es una alternativa y cada fila una observación.
    Devuelve la figura matplotlib.
    """
    fig, ax = plt.subplots()
    # Promedio por alternativa
    mean_probs = probs_df.mean(axis=0)
    ax.bar(mean_probs.index, mean_probs.values)
    ax.set_xlabel("Alternativa")
    ax.set_ylabel("Probabilidad media")
    ax.set_title("Distribución de probabilidades de elección")
    if show:
        plt.show()
    return fig

def plot_elasticities(elasticities: dict, show=False):
    """
    Dibuja un gráfico de barras con las elasticidades calculadas.
    - elasticities: dict {variable: elasticidad}
    Devuelve la figura matplotlib.
    """
    fig, ax = plt.subplots()
    vars_ = list(elasticities.keys())
    vals = list(elasticities.values())
    ax.bar(vars_, vals)
    ax.set_xlabel("Variable")
    ax.set_ylabel("Elasticidad")
    ax.set_title("Elasticidades estimadas")
    if show:
        plt.show()
    return fig

def plot_rational_path(tracker_json, show=False):
    """
    Dibuja un diagrama simple de la secuencia de preguntas y respuestas.
    - tracker_json: {'steps': [ {'question','answer',...}, ... ]}
    Devuelve la figura matplotlib.
    """
    steps = tracker_json.get("steps", [])
    questions = [s["question"] for s in steps]
    answers = [s["answer"] or "<sin respuesta>" for s in steps]
    y = np.arange(len(steps))
    
    fig, ax = plt.subplots(figsize=(6, len(steps)*0.5))
    ax.scatter([0]*len(steps), y)
    ax.hlines(y, xmin=0, xmax=1)
    for i, (q, a) in enumerate(zip(questions, answers)):
        ax.text(0.05, i, f"Q: {q}", va='center')
        ax.text(0.5, i, f"A: {a}", va='center')
    ax.axis('off')
    ax.set_title("Camino racional: preguntas y respuestas")
    if show:
        plt.show()
    return fig
