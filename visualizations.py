import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def graficar_elasticidades(elasticidades: dict):
    """
    Genera un gráfico de barras interactivo para elasticidades.

    Parámetros:
        elasticidades (dict): {variable: valor_elasticidad}
    """
    if not elasticidades:
        return None

    df_elas = pd.DataFrame(list(elasticidades.items()), columns=["Variable", "Elasticidad"])
    fig = px.bar(
        df_elas,
        x="Variable",
        y="Elasticidad",
        title="Elasticidades de las variables",
        text="Elasticidad",
        labels={"Elasticidad": "Elasticidad", "Variable": "Variable"},
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        yaxis=dict(title="Elasticidad"),
        xaxis=dict(title="Variable"),
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        margin=dict(t=40, b=40)
    )
    return fig

def graficar_simulacion_contrafactual(pred_original: float, pred_contrafactual: float):
    """
    Genera gráfico comparativo de barras para predicción original vs contrafactual.

    Parámetros:
        pred_original (float): Valor predicho con datos originales.
        pred_contrafactual (float): Valor predicho con escenario contrafactual.
    """
    fig = go.Figure(
        data=[
            go.Bar(name='Original', x=['Predicción'], y=[pred_original], marker_color='blue'),
            go.Bar(name='Contrafactual', x=['Predicción'], y=[pred_contrafactual], marker_color='orange')
        ]
    )
    fig.update_layout(
        title="Comparación de predicción original y contrafactual",
        yaxis_title="Valor predicho",
        barmode='group',
        margin=dict(t=40, b=40)
    )
    return fig
