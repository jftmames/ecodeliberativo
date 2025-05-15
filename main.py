import streamlit as st
from econometrics import *
from deliberation_engine import *
from navigator import *
from epistemic_metrics import *
from report_generator import *
from ui import *

def main():
    st.title("Simulador Econométrico-Deliberativo")

    # Selección de modo de uso
    modo = st.sidebar.selectbox("Modo de uso", ["Docente", "Consultor", "Institucional"])
    # Carga de datos (desde ejemplo.csv o subida por el usuario)
    st.sidebar.header("Datos de análisis")
    data_source = st.sidebar.radio("¿Cómo quieres cargar los datos?", ["Ejemplo", "Subir CSV"])
    if data_source == "Ejemplo":
        import pandas as pd
        df = pd.read_csv("data/ejemplo.csv")
    else:
        uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type="csv")
        if uploaded_file is not None:
            import pandas as pd
            df = pd.read_csv(uploaded_file)
        else:
            st.stop()

    # Selección de modelo econométrico
    modelo = st.sidebar.selectbox("Modelo econométrico", [
        "Logit", "Probit", "Tobit", "MNL", "OLS", "Poisson"
    ])

    # Aquí comenzaría el flujo deliberativo (a integrar)
    st.write(f"Modelo seleccionado: {modelo}")
    st.dataframe(df.head())

    # (Aquí se integrarán las funciones deliberativas y visualizaciones)
    # Ejemplo:
    # resultado = ejecutar_modelo_deliberativo(df, modelo)
    # mostrar_informe(resultado)

if __name__ == "__main__":
    main()
