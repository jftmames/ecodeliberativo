# ui.py

import streamlit as st
import statsmodels.api as sm
import pandas as pd

from data_loader import generate_synthetic_data, load_csv
from econometrics import fit_logit, fit_ols, summarize_model, compute_average_probability
from mnl import fit_mnl, summarize_mnl, predict_mnl
from elasticities import compute_elasticities
from validation import check_model_diagnostics
from visualizations import plot_choice_probabilities, plot_elasticities
from epistemic_metrics import compute_eee
from navigator import display_navigation
from deliberation_engine import InquiryEngine, ReasoningTracker
from report_generator import build_report

# --- Configuración de la página ---
st.set_page_config(page_title="Simulador Econ-Delib", layout="wide")
st.title("Simulador Econométrico-Deliberativo")
st.markdown("MVP modular para universidades y consultores.")

# Variables globales
FEATURES = ["precio", "ingreso", "edad"]

# Inicializar session_state
for key, default in [
    ("df", None),
    ("model", None),
    ("model_type", None),
    ("probs_df", None),
    ("diagnostics", None),
    ("elasticities", None),
    ("subquestions", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default
if "engine" not in st.session_state:
    st.session_state.engine = InquiryEngine()
if "tracker" not in st.session_state:
    st.session_state.tracker = ReasoningTracker()

# Sidebar: fases
phase = st.sidebar.radio("Fase:", [
    "1. Datos",
    "2. Econometría",
    "3. Deliberación",
    "4. Informe"
])

# -------------------------
# Fase 1: Datos
# -------------------------
if phase == "1. Datos":
    st.header("1. Datos de ejemplo / carga CSV")
    source = st.radio("Origen de datos", ["Ejemplo", "Subir CSV"])
    if source == "Ejemplo":
        st.session_state.df = generate_synthetic_data()
    else:
        uploaded = st.file_uploader("Sube un CSV", type="csv")
        if uploaded:
            st.session_state.df = load_csv(uploaded)
        else:
            st.stop()
    st.write("Vista previa de datos:")
    st.dataframe(st.session_state.df.head())
    st.write(f"Total registros: **{len(st.session_state.df)}**")

# -------------------------
# Fase 2: Econometría
# -------------------------
elif phase == "2. Econometría":
    st.header("2. Módulo Econométrico")
    if st.session_state.df is None:
        st.warning("Completa primero la Fase 1 (Datos).")
        st.stop()
    df = st.session_state.df
    st.dataframe(df.head())

    st.session_state.model_type = st.selectbox("Modelo econométrico", ["Logit", "OLS", "MNL"])
    if st.button("Ajustar modelo"):
        # Ajuste
        if st.session_state.model_type == "Logit":
            model = fit_logit(df, FEATURES, "eleccion")
            st.session_state.probs_df = None
        elif st.session_state.model_type == "OLS":
            model = fit_ols(df, FEATURES, "eleccion")
            st.session_state.probs_df = None
        else:
            model = fit_mnl(df, FEATURES, "eleccion")
            st.session_state.probs_df = predict_mnl(model, df, FEATURES)
        st.session_state.model = model

        # Mostrar resumen
        summary = (
            summarize_model(model)
            if st.session_state.model_type != "MNL"
            else summarize_mnl(model)
        )
        st.subheader("Resumen del modelo")
        st.text(summary)

        # Gráficas de probabilidades
        if st.session_state.model_type == "Logit":
            probs_df = pd.DataFrame({"prob": model.predict(sm.add_constant(df[FEATURES]))})
            fig = plot_choice_probabilities(probs_df)
            st.pyplot(fig)
        elif st.session_state.model_type == "MNL":
            fig = plot_choice_probabilities(st.session_state.probs_df)
            st.pyplot(fig)

        # Elasticidades (solo Logit)
        if st.session_state.model_type == "Logit":
            elas = compute_elasticities(
                model, df, FEATURES, "Logit"
            )
            st.session_state.elasticities = elas
            fig2 = plot_elasticities(elas)
            st.subheader("Elasticidades (Logit)")
            st.pyplot(fig2)

        # Diagnósticos
        diag = check_model_diagnostics(df, model, FEATURES)
        st.session_state.diagnostics = diag
        st.subheader("Diagnósticos econométricos")
        st.json(diag)

# -------------------------
# Fase 3: Deliberación
# -------------------------
elif phase == "3. Deliberación":
    st.header("3. Módulo de Deliberación")
    prompt = st.text_input("Describe el análisis que quieres realizar:")
    if st.button("Generar subpreguntas"):
        subs = st.session_state.engine.generate_subquestions(prompt)
        st.session_state.subquestions = subs
        st.session_state.tracker = ReasoningTracker()

    if st.session_state.subquestions:
        st.markdown("**Responde a cada subpregunta:**")
        for i, q in enumerate(st.session_state.subquestions):
            st.text_area(label=f"{i+1}. {q}", key=f"ans_{i}", height=80)
        if st.button("Registrar respuestas y calcular EEE"):
            for i, q in enumerate(st.session_state.subquestions):
                ans = st.session_state.get(f"ans_{i}", "").strip()
                st.session_state.tracker.add_step(q, ans)
            eee_metrics = compute_eee(
                st.session_state.tracker.to_json(),
                max_steps=len(st.session_state.subquestions)
            )
            st.subheader("Registro de razonamiento")
            st.json(st.session_state.tracker.to_json())
            st.subheader("Métricas epistémicas (EEE)")
            st.json(eee_metrics)
            display_navigation(st.session_state.tracker.to_json())

# -------------------------
# Fase 4: Informe
# -------------------------
elif phase == "4. Informe":
    st.header("4. Generar Informe")
    if st.session_state.df is None or st.session_state.model is None:
        st.warning("Completa las fases 1 y 2 antes de generar el informe.")
        st.stop()
    data_summary = (
        f"Dataset: {len(st.session_state.df)} filas, variables: "
        f"{', '.join(st.session_state.df.columns)}"
    )
    model_name = st.session_state.model_type
    model_summary = (
        summarize_model(st.session_state.model)
        if model_name != "MNL"
        else summarize_mnl(st.session_state.model)
    )
    model_summaries = {model_name: model_summary}
    reasoning_log = (
        st.session_state.tracker.to_json()
        if st.session_state.tracker.log else {}
    )
    html = build_report(data_summary, model_summaries, reasoning_log)
    st.download_button("Descargar informe (HTML)", html, file_name="informe.html")

# Pie de página
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Simulador Econ-Delib")

