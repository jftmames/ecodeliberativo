import streamlit as st
import pandas as pd
from econometrics import run_model
from deliberation_engine import preguntar_deliberativo
from report_generator import generar_informe_html
from epistemic_metrics import calcular_metricas_deliberativas, perfil_eee

import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Simulador Econométrico-Deliberativo", layout="wide")

def mostrar_tips_perfil(modo):
    if modo == "Docente":
        st.sidebar.markdown(
            """
            ### Tips para modo Docente
            - Explique a sus estudiantes la importancia de cada variable en el modelo.
            - Use las preguntas deliberativas para fomentar el pensamiento crítico.
            - Aproveche las simulaciones para mostrar escenarios contrafactuales.
            """
        )
    elif modo == "Consultor":
        st.sidebar.markdown(
            """
            ### Tips para modo Consultor
            - Analice los coeficientes para identificar variables clave para sus clientes.
            - Use los diagnósticos para validar la calidad del modelo.
            - Prepare informes descargables para sus presentaciones.
            """
        )
    elif modo == "Institucional":
        st.sidebar.markdown(
            """
            ### Tips para modo Institucional
            - Enfóquese en la interpretación estratégica de las elasticidades.
            - Use los índices EEE para evaluar la calidad de la toma de decisiones.
            - Realice comparativas de modelos para elegir la mejor política.
            """
        )

def main():
    st.title("Simulador Econométrico-Deliberativo")

    modo = st.sidebar.selectbox("Modo de uso", ["Docente", "Consultor", "Institucional"])
    mostrar_tips_perfil(modo)

    st.sidebar.header("Datos de análisis")
    data_source = st.sidebar.radio("¿Cómo quieres cargar los datos?", ["Ejemplo", "Subir CSV"])
    if data_source == "Ejemplo":
        try:
            df = pd.read_csv("data/ejemplo.csv")
        except FileNotFoundError:
            st.error("No se encontró el archivo de ejemplo en la ruta 'data/ejemplo.csv'.")
            st.stop()
    else:
        uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("Sube un archivo CSV para continuar.")
            st.stop()

    st.write("Vista previa de los datos:")
    st.dataframe(df.head(10))

    st.sidebar.header("Variables")
    y_var = st.sidebar.selectbox("Variable dependiente", df.columns)
    x_vars = st.sidebar.multiselect("Variables independientes", [col for col in df.columns if col != y_var])

    if not x_vars:
        st.info("Selecciona al menos una variable independiente para continuar.")
        st.stop()

    modelos_disponibles = ["Logit", "Probit", "Tobit", "MNL", "OLS", "Poisson"]
    st.sidebar.header("Modelos a analizar")
    modelos_seleccionados = st.sidebar.multiselect("Selecciona uno o varios modelos", modelos_disponibles, default=["Logit"])

    if "Tobit" in modelos_seleccionados:
        st.warning("El modelo Tobit está temporalmente deshabilitado por incompatibilidad con la versión actual de Python.")
        modelos_seleccionados = [m for m in modelos_seleccionados if m != "Tobit"]
        if not modelos_seleccionados:
            st.stop()

    params_dict = {modelo: {} for modelo in modelos_seleccionados}

    if st.sidebar.button("Ejecutar modelos seleccionados"):
        resultados_modelos = {}
        aics = {}
        bics = {}
        summaries = {}
        st.session_state["deliberation_answers"] = {}
        for modelo in modelos_seleccionados:
            try:
                res = run_model(df, modelo, y_var, x_vars, **params_dict.get(modelo, {}))
                resultados_modelos[modelo] = res
                aics[modelo] = res["diagnostics"].get("AIC", None)
                bics[modelo] = res["diagnostics"].get("BIC", None)
                summaries[modelo] = res["summary"]
                st.session_state["resultados_modelos"] = resultados_modelos
                st.session_state["aics"] = aics
                st.session_state["bics"] = bics
                st.session_state["summaries"] = summaries
            except Exception as e:
                st.error(f"Error en modelo {modelo}: {e}")
    else:
        resultados_modelos = st.session_state.get("resultados_modelos", {})
        aics = st.session_state.get("aics", {})
        bics = st.session_state.get("bics", {})
        summaries = st.session_state.get("summaries", {})

    modelo_ref = None
    if resultados_modelos:
        for m in modelos_seleccionados:
            if m in resultados_modelos:
                modelo_ref = m
                break

    tab1, tab2, tab3, tab4 = st.tabs(["Análisis", "Deliberación", "Simulación", "Resultados e Informe"])

    with tab1:
        st.header("1️⃣ Dashboard de Análisis y Visualización")
        if not modelo_ref:
            st.info("Ejecuta primero los modelos para ver análisis.")
        else:
            res = resultados_modelos[modelo_ref]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nº observaciones", len(df))
            with col2:
                st.metric("Nº variables", len(x_vars))
            with col3:
                st.metric("Modelo principal", modelo_ref)

            if hasattr(res["coef"], "index"):
                coef_df = res["coef"].to_frame("Coeficiente").reset_index().rename(columns={"index": "Variable"})
                fig_coef = px.bar(coef_df, x="Variable", y="Coeficiente", title="Coeficientes del modelo", text_auto=True)
                st.plotly_chart(fig_coef, use_container_width=True)

            if "predicted" in res:
                pred = res["predicted"]
                st.markdown("#### Distribución de predicciones")
                fig_pred = px.histogram(pred, nbins=20, title="Distribución de predicciones")
                st.plotly_chart(fig_pred, use_container_width=True)

            if "elasticities" in res:
                st.markdown("#### Elasticidades")
                elas = res["elasticities"]
                if isinstance(elas, dict):
                    elas_df = pd.DataFrame(list(elas.items()), columns=["Variable", "Elasticidad"])
                    fig_elas = px.bar(elas_df, x="Variable", y="Elasticidad", title="Elasticidades")
                    st.plotly_chart(fig_elas, use_container_width=True)

            if len(resultados_modelos) > 1:
                st.markdown("#### Comparativa de modelos (AIC/BIC)")
                comp_df = pd.DataFrame({
                    "Modelo": list(aics.keys()),
                    "AIC": [aics[m] for m in aics],
                    "BIC": [bics[m] for m in bics],
                })
                st.dataframe(comp_df)
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(x=comp_df["Modelo"], y=comp_df["AIC"], name="AIC"))
                fig_comp.add_trace(go.Bar(x=comp_df["Modelo"], y=comp_df["BIC"], name="BIC"))
                fig_comp.update_layout(barmode='group', title="Comparativa AIC/BIC")
                st.plotly_chart(fig_comp, use_container_width=True)

    with tab2:
        st.header("2️⃣ Deliberación estructurada y feedback")
        if not modelo_ref:
            st.info("Ejecuta primero un modelo y responde a las preguntas para deliberar.")
        else:
            res = resultados_modelos[modelo_ref]

            st.subheader("Preguntas para el razonamiento deliberativo")
            respuestas_usuario = preguntar_deliberativo(res["questions"])

            st.subheader("Tus respuestas deliberativas")
            for i, (preg, resp) in enumerate(respuestas_usuario.items(), 1):
                st.markdown(f"**{i}. {preg}**<br>{resp}", unsafe_allow_html=True)

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

            metricas = calcular_metricas_deliberativas(respuestas_usuario)
            st.subheader("Métricas deliberativas detalladas")
            st.metric("EEE (Índice de Equilibrio Erotético)", metricas['EEE'])
            st.metric("Coherencia", metricas['Coherencia'])
            st.metric("Profundidad (palabras promedio)", metricas['Profundidad'])
            st.metric("Cobertura (proporción respuestas)", metricas['Cobertura'])
            st.metric("Exploración", metricas['Exploración'])
            st.caption(perfil_eee(metricas['EEE']))

            st.markdown("### 🗣️ ¿Qué interpretación sugieres a partir de este resultado?")
            feedback_user = st.text_area("Tu interpretación/reflexión:", key="feedback_user")
            if st.button("Guardar interpretación", key="btn_feedback"):
                st.success("¡Interpretación guardada! Se añadirá al informe final.")

            delib_df = pd.DataFrame(list(respuestas_usuario.items()), columns=["Pregunta", "Respuesta"])
            st.download_button(
                label="Descargar respuestas deliberativas (CSV)",
                data=delib_df.to_csv(index=False).encode("utf-8"),
                file_name="respuestas_deliberativas.csv",
                mime="text/csv"
            )

    with tab3:
        st.header("3️⃣ Simulación contrafactual")
        if not modelo_ref:
            st.info("Ejecuta primero un modelo para simular escenarios contrafactuales.")
        else:
            res = resultados_modelos[modelo_ref]

            st.write("Ajusta valores para simular un escenario contrafactual:")

            contrafactual = {}
            for var in x_vars:
                min_v = float(df[var].min())
                max_v = float(df[var].max())
                mean_v = float(df[var].mean())
                contrafactual[var] = st.slider(f"{var}", min_value=min_v, max_value=max_v, value=mean_v)

            row = pd.DataFrame([contrafactual])
            if "const" in res["coef"].index:
                row.insert(0, "const", 1.0)
            try:
                if "result" in res:
                    pred_cf = res["result"].predict(row)[0]
                    st.success(f"Predicción contrafactual: {pred_cf:.4f}")

                    original_pred = res["result"].predict(res["result"].model.exog)[0]
                    fig_cf = go.Figure(data=[
                        go.Bar(name='Original', x=['Predicción'], y=[original_pred]),
                        go.Bar(name='Contrafactual', x=['Predicción'], y=[pred_cf])
                    ])
                    fig_cf.update_layout(barmode='group')
                    st.plotly_chart(fig_cf, use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudo calcular la predicción contrafactual: {e}")

    with tab4:
        st.header("4️⃣ Resultados completos, diagnósticos y exportación de informe")
        if not modelo_ref:
            st.info("Ejecuta primero un modelo para ver y descargar resultados.")
        else:
            res = resultados_modelos[modelo_ref]
            st.subheader("Resumen del modelo principal")
            st.text(res["summary"])

            st.subheader("Coeficientes del modelo")
            st.dataframe(res["coef"].to_frame(name="Coeficiente"))

            st.subheader("Diagnóstico automático")
            st.json(res["diagnostics"])

            st.download_button(
                label="Descargar coeficientes como CSV",
                data=res["coef"].to_csv().encode("utf-8"),
                file_name=f"coeficientes_{modelo_ref.lower()}.csv",
                mime="text/csv"
            )
            diag_df = pd.DataFrame([res["diagnostics"]])
            st.download_button(
                label="Descargar diagnóstico como CSV",
                data=diag_df.to_csv(index=False).encode("utf-8"),
                file_name=f"diagnostico_{modelo_ref.lower()}.csv",
                mime="text/csv"
            )

            st.markdown("---")
            st.subheader("Informe deliberativo")
            respuestas_usuario = st.session_state.get("deliberation_answers", {})
            feedback_user = st.session_state.get("feedback_user", "")
            eee = metricas['EEE'] if 'metricas' in locals() else None

            if st.button("Generar informe HTML", key="gen_html"):
                informe_html = generar_informe_html(
                    modo=modo,
                    modelo=modelo_ref,
                    y_var=y_var,
                    x_vars=x_vars,
                    resumen=res["summary"],
                    coef=res["coef"],
                    diagnostico=res["diagnostics"],
                    deliberacion=list(respuestas_usuario.items()) if respuestas_usuario else [],
                    eee=eee,
                    eee_texto=perfil_eee(eee) if eee else "",
                    feedback=feedback_user
                )
                st.components.v1.html(informe_html, height=800, scrolling=True)
                st.download_button(
                    label="Descargar informe HTML",
                    data=informe_html.encode("utf-8"),
                    file_name="informe_deliberativo.html",
                    mime="text/html"
                )

if __name__ == "__main__":
    main()
