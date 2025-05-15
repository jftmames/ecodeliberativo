import streamlit as st
import pandas as pd
from econometrics import run_model

def main():
    st.title("Simulador Econométrico-Deliberativo")

    # Selección de modo de uso (por ahora, solo informativo; futuro: adapta UI/respuestas)
    modo = st.sidebar.selectbox("Modo de uso", ["Docente", "Consultor", "Institucional"])

    # Carga de datos
    st.sidebar.header("Datos de análisis")
    data_source = st.sidebar.radio("¿Cómo quieres cargar los datos?", ["Ejemplo", "Subir CSV"])
    if data_source == "Ejemplo":
        df = pd.read_csv("data/ejemplo.csv")
    else:
        uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("Sube un archivo CSV para continuar.")
            st.stop()

    st.write("Vista previa de los datos:")
    st.dataframe(df.head(10))

    # Selección de variables
    st.sidebar.header("Variables")
    y_var = st.sidebar.selectbox("Variable dependiente", df.columns)
    x_vars = st.sidebar.multiselect("Variables independientes", [col for col in df.columns if col != y_var])

    if not x_vars:
        st.info("Selecciona al menos una variable independiente para continuar.")
        st.stop()

    # Selección de modelo econométrico
    modelo = st.sidebar.selectbox("Modelo econométrico", [
        "Logit", "Probit", "Tobit", "MNL", "OLS", "Poisson"
    ])

    # Parámetros extra (ejemplo: Tobit)
    params = {}
    if modelo == "Tobit":
        st.sidebar.markdown("**Parámetros Tobit**")
        params["left"] = st.sidebar.number_input("Límite inferior (left)", value=0.0)
        params["right"] = st.sidebar.number_input("Límite superior (right)", value=float("inf"))

    if st.button("Ejecutar modelo"):
        try:
            result_dict = run_model(df, modelo, y_var, x_vars, **params)
            st.success(f"Modelo {modelo} ejecutado correctamente.")

            st.subheader("Resumen del modelo")
            st.text(result_dict["summary"])

            st.subheader("Coeficientes del modelo")
            st.dataframe(result_dict["coef"].to_frame(name="Coeficiente"))

            st.subheader("Preguntas para el razonamiento deliberativo")
            for i, question in enumerate(result_dict["questions"], 1):
                st.markdown(f"**{i}.** {question}")

            st.subheader("Diagnóstico automático del modelo")
            st.json(result_dict["diagnostics"])

            # Descarga de coeficientes
            st.download_button(
                label="Descargar coeficientes como CSV",
                data=result_dict["coef"].to_csv().encode("utf-8"),
                file_name=f"coeficientes_{modelo.lower()}.csv",
                mime="text/csv"
            )
            # Descarga de diagnósticos
            diag_df = pd.DataFrame([result_dict["diagnostics"]])
            st.download_button(
                label="Descargar diagnóstico como CSV",
                data=diag_df.to_csv(index=False).encode("utf-8"),
                file_name=f"diagnostico_{modelo.lower()}.csv",
                mime="text/csv"
            )

            # (Próximo paso: integración de deliberation_engine para capturar respuestas)
        except Exception as e:
            st.error(f"Error al ejecutar el modelo: {e}")
    else:
        st.info("Configura los parámetros y haz click en 'Ejecutar modelo'.")

if __name__ == "__main__":
    main()
