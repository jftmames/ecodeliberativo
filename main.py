import streamlit as st
import pandas as pd
from econometrics import run_model
from deliberation_engine import preguntar_deliberativo
from report_generator import generar_informe_html
from epistemic_metrics import calcular_eee, perfil_eee

import networkx as nx
import matplotlib.pyplot as plt

def main():
    st.title("Simulador Econométrico-Deliberativo")

    # 1. Selección de modo de uso
    modo = st.sidebar.selectbox("Modo de uso", ["Docente", "Consultor", "Institucional"])

    # 2. Carga de datos
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

    # 3. Selección de variables
    st.sidebar.header("Variables")
    y_var = st.sidebar.selectbox("Variable dependiente", df.columns)
    x_vars = st.sidebar.multiselect("Variables independientes", [col for col in df.columns if col != y_var])

    if not x_vars:
        st.info("Selecciona al menos una variable independiente para continuar.")
        st.stop()

    # 4. Selección de modelo econométrico
    modelo = st.sidebar.selectbox("Modelo econométrico", [
        "Logit", "Probit", "Tobit", "MNL", "OLS", "Poisson"
    ])

    # 5. Parámetros extra (Tobit)
    params = {}
    if modelo == "Tobit":
        st.sidebar.markdown("**Parámetros Tobit**")
        params["left"] = st.sidebar.number_input("Límite inferior (left)", value=0.0)
        params["right"] = st.sidebar.number_input("Límite superior (right)", value=float("inf"))

    # 6. Ejecución del modelo y deliberación
    if st.button("Ejecutar modelo"):
        try:
            result_dict = run_model(df, modelo, y_var, x_vars, **params)
            st.success(f"Modelo {modelo} ejecutado correctamente.")

            st.subheader("Resumen del modelo")
            st.text(result_dict["summary"])

            st.subheader("Coeficientes del modelo")
            st.dataframe(result_dict["coef"].to_frame(name="Coeficiente"))

            # Preguntas y respuestas deliberativas
            st.subheader("Preguntas para el razonamiento deliberativo")
            respuestas_usuario = preguntar_deliberativo(result_dict["questions"])

            st.subheader("Tus respuestas deliberativas")
            for i, (preg, resp) in enumerate(respuestas_usuario.items(), 1):
                st.markdown(f"**{i}. {preg}**<br>{resp}", unsafe_allow_html=True)

            # --- Visualización de árbol epistémico ---
            st.subheader("Árbol epistémico (trazabilidad del razonamiento)")
            try:
                G = nx.DiGraph()
                root = "Pregunta raíz"
                G.add_node(root)
                for i, (preg, resp) in enumerate(respuestas_usuario.items(), 1):
                    nodo_p = f"Q{i}: {preg}"
                    G.add_node(nodo_p)
                    G.add_edge(root, nodo_p)
                    if resp.strip():
                        nodo_r = f"R{i}: {resp[:30]}..." if len(resp) > 30 else f"R{i}: {resp}"
                        G.add_node(nodo_r)
                        G.add_edge(nodo_p, nodo_r)
                fig, ax = plt.subplots(figsize=(8, 2 + len(respuestas_usuario)//2))
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=True, node_color="#c6dbef", font_size=8, ax=ax)
                st.pyplot(fig)
            except Exception as ex:
                st.info("No se pudo dibujar el árbol epistémico. Error: " + str(ex))

            # --- Cálculo y visualización del Índice de Equilibrio Erotético (EEE) ---
            eee = calcular_eee(respuestas_usuario)
            st.subheader("Índice de Equilibrio Erotético (EEE)")
            st.metric("EEE", eee)
            st.caption(perfil_eee(eee))

            # Descarga de respuestas deliberativas
            delib_df = pd.DataFrame(list(respuestas_usuario.items()), columns=["Pregunta", "Respuesta"])
            st.download_button(
                label="Descargar respuestas deliberativas (CSV)",
                data=delib_df.to_csv(index=False).encode("utf-8"),
                file_name="respuestas_deliberativas.csv",
                mime="text/csv"
            )

            # Diagnóstico automático
            st.subheader("Diagnóstico automático del modelo")
            st.json(result_dict["diagnostics"])

            # Descarga de coeficientes y diagnóstico
            st.download_button(
                label="Descargar coeficientes como CSV",
                data=result_dict["coef"].to_csv().encode("utf-8"),
                file_name=f"coeficientes_{modelo.lower()}.csv",
                mime="text/csv"
            )
            diag_df = pd.DataFrame([result_dict["diagnostics"]])
            st.download_button(
                label="Descargar diagnóstico como CSV",
                data=diag_df.to_csv(index=False).encode("utf-8"),
                file_name=f"diagnostico_{modelo.lower()}.csv",
                mime="text/csv"
            )

            # --- Generación y visualización de informe HTML ---
            st.markdown("---")
            st.subheader("Informe deliberativo")
            if st.button("Generar informe HTML"):
                informe_html = generar_informe_html(
                    modo=modo,
                    modelo=modelo,
                    y_var=y_var,
                    x_vars=x_vars,
                    resumen=result_dict["summary"],
                    coef=result_dict["coef"],
                    diagnostico=result_dict["diagnostics"],
                    deliberacion=list(respuestas_usuario.items()),
                    eee=eee,
                    eee_texto=perfil_eee(eee)
                )
                st.components.v1.html(informe_html, height=800, scrolling=True)
                st.download_button(
                    label="Descargar informe HTML",
                    data=informe_html.encode("utf-8"),
                    file_name="informe_deliberativo.html",
                    mime="text/html"
                )
        except Exception as e:
            st.error(f"Error al ejecutar el modelo: {e}")
    else:
        st.info("Configura los parámetros y haz click en 'Ejecutar modelo'.")

if __name__ == "__main__":
    main()
