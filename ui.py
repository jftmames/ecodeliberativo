# ui.py
import streamlit as st

# Importar los módulos especializados
from econometria import run_econometria
from deliberacion import run_deliberacion
from diagnostico import run_diagnostico
from informe import run_informe
from data import load_data

def main():
    st.set_page_config(page_title="Simulador Econométrico-Deliberativo", layout="wide")
    st.title("Simulador Econométrico-Deliberativo para Decisiones de Consumo")

    # Cargar datos
    df, FEATURES = load_data()

    # Si no hay variables seleccionadas, terminamos
    if not FEATURES:
        return

    # Crear pestañas
    tabs = st.tabs([
        "1. Datos",
        "2. Econometría",
        "3. Deliberación",
        "4. Diagnóstico",
        "5. Informe"
    ])

    # 1. Datos (ya handled en load_data, quizás muestra un resumen)
    with tabs[0]:
        st.header("1. Datos")
        st.write(df.head())

    # 2. Econometría
    with tabs[1]:
        st.header("2. Econometría")
        run_econometria(df, FEATURES)

    # 3. Deliberación
    with tabs[2]:
        st.header("3. Deliberación epistémica")
        run_deliberacion(df, FEATURES)

    # 4. Diagnóstico
    with tabs[3]:
        st.header("4. Diagnóstico del modelo")
        run_diagnostico(df, FEATURES)

    # 5. Informe
    with tabs[4]:
        st.header("5. Informe final")
        run_informe(df, FEATURES)

if __name__ == "__main__":
    main()

