# ui.py
import streamlit as st
from data_loader import generate_synthetic_data, load_csv
from econometrics import fit_logit, fit_ols, summarize_model, compute_average_probability
from deliberation_engine import InquiryEngine, ReasoningTracker
from report_generator import build_report, export_pdf

# Configuración de la página
st.set_page_config(page_title="Simulador Econ-Delib", layout="wide")
st.title("Simulador Econométrico-Deliberativo")
st.markdown("MVP modular para universidades y consultores.")

# Sidebar: Navegación
phase = st.sidebar.radio("Fase:", [
    "1. Datos",
    "2. Econometría",
    "3. Deliberación",
    "4. Informe"
])

# Fase 1: Datos
if phase == "1. Datos":
    st.header("1. Datos de ejemplo / carga CSV")
    source = st.radio("Origen de datos", ["Ejemplo", "Subir CSV"])
    if source == "Ejemplo":
        df = generate_synthetic_data()
    else:
        uploaded = st.file_uploader("Sube un CSV", type="csv")
        if uploaded:
            df = load_csv(uploaded)
        else:
            st.stop()
    st.dataframe(df.head())
    st.write(f"Total registros: {len(df)}")

# Fase 2: Econometría
elif phase == "2. Econometría":
    st.header("2. Módulo Econométrico")
    df = generate_synthetic_data()  # o reutilizar global
    st.dataframe(df.head())
    model_type = st.selectbox("Modelo", ["Logit", "OLS"])
    if st.button("Ejecutar"):
        features = ["precio", "ingreso", "edad"]
        if model_type == "Logit":
            model = fit_logit(df, features, "eleccion")
        else:
            model = fit_ols(df, features, "eleccion")
        summary = summarize_model(model)
        st.text(summary)
        if model_type == "Logit":
            prob = compute_average_probability(model, df, features)
            st.markdown(f"Probabilidad media: **{prob:.2f}**")

# Fase 3: Deliberación
elif phase == "3. Deliberación":
    st.header("3. Módulo de Deliberación (proto)")
    engine = InquiryEngine()
    tracker = ReasoningTracker()
    prompt = st.text_input("Describe el problema:")
    if st.button("Generar subpreguntas"):
        subs = engine.generate_subquestions(prompt)
        for q in subs:
            tracker.add_step(q, answer="", metadata={})
        st.json(tracker.to_json())

# Fase 4: Informe
elif phase == "4. Informe":
    st.header("4. Generar Informe")
    # Recoge resúmenes ficticios para la demo
    data_summary = "Datos simulados: 200 filas, 4 columnas."
    model_summaries = {"Logit": "Resumen Logit..."}
    reasoning_log = {"steps": []}
    html = build_report(data_summary, model_summaries, reasoning_log)
    st.download_button("Descargar informe (HTML)", html, file_name="informe.html")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Simulador Econ-Delib")
