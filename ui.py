# ui.py
import streamlit as st
import pandas as pd
import numpy as np

from mnl import fit_mnl, predict_mnl
from elasticities import compute_elasticities
from deliberation_engine import DeliberationEngine
from navigator import EpistemicNavigator
from validation import check_model_diagnostics
from report_generator import build_report, export_pdf
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS

def load_example_data():
    # Simula un dataset de ejemplo
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "precio": np.random.uniform(1, 10, n),
        "ingreso": np.random.uniform(20, 100, n),
        "edad": np.random.randint(18, 70, n),
    })
    # Genera Y binaria con un logit simple
    X = np.column_stack([np.ones(n), df["precio"], df["ingreso"], df["edad"]])
    beta = np.array([-1.0, -0.5, 0.03, 0.01])
    logits = 1 / (1 + np.exp(-X.dot(beta)))
    df["Y"] = np.random.binomial(1, logits)
    return df

def fit_logit(df, features):
    X = df[features]
    X = st.session_state.const_column = sm.add_constant(X)
    y = df["Y"]
    model = Logit(y, X).fit(disp=False)
    return model

def main():
    st.set_page_config(page_title="Simulador Econométrico-Deliberativo", layout="wide")
    st.title("Simulador Econométrico-Deliberativo para Decisiones de Consumo")

    # --- Selector de rol ---
    role = st.sidebar.selectbox("Modo de uso", ["Docente", "Consultor"])

    # --- 1. Datos ---
    st.header("1. Datos")
    uploaded = st.file_uploader("Sube un CSV con tus datos (debe incluir Y = elección)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.info("Usando datos de ejemplo.")
        df = load_example_data()

    st.write("Primeras filas de los datos:", df.head())

    # --- Selección de variables ---
    FEATURES = st.multiselect("Selecciona variables explicativas:", [c for c in df.columns if c != "Y"])
    if not FEATURES:
        st.warning("Debes seleccionar al menos una variable explicativa.")
        st.stop()

    # --- 2. Econometría ---
    st.header("2. Econometría")
    model_type = st.radio("Elige modelo:", ["Logit", "OLS", "MNL"])

    if model_type == "Logit":
        # Ajuste
        model = fit_logit(df, FEATURES)

        # Mostrar fórmula Logit
        st.markdown("### Modelo Logit estimado")
        terms = " + ".join([f"β₍{i+1}₎·{FEATURES[i]}" for i in range(len(FEATURES))])
        formula = f"P(Y=1|X) = 1 / (1 + exp(-[β₀ + {terms}]))"
        st.latex(formula)

        # Pasos de estimación
        with st.expander("Ver pasos de estimación"):
            st.write("1. Se crea la matriz de regresores X (incluye constante).")
            st.write(f"2. Variables: {', '.join(FEATURES)}.")
            st.write("3. Se ajusta por máxima verosimilitud el modelo Logit.")

        # Tabla de coeficientes e interpretación
        coefs = model.params.reset_index()
        coefs.columns = ["Variable", "Coeficiente"]
        coefs["p-valor"] = model.pvalues.values
        interpretaciones = []
        for coef in coefs["Coeficiente"]:
            signo = "incrementa" if coef > 0 else "reduce"
            interpretaciones.append(f"A {signo} la probabilidad de elección.")
        coefs["Interpretación"] = interpretaciones

        st.markdown("#### Resultados del ajuste")
        st.dataframe(coefs)

        # Elasticidades
        st.subheader("Elasticidades")
        elasts = compute_elasticities(model, df, FEATURES)
        st.table(elasts)

    elif model_type == "OLS":
        # Ajuste OLS
        X = sm.add_constant(df[FEATURES])
        y = df["Y"]
        model = OLS(y, X).fit()

        st.markdown("### Modelo OLS estimado")
        terms = " + ".join(FEATURES)
        formula = f"Y = β₀ + {terms}"
        st.latex(formula)

        with st.expander("Ver pasos de estimación"):
            st.write("1. Matriz X con constante.")
            st.write(f"2. Variables: {', '.join(FEATURES)}.")
            st.write("3. Ajuste por Mínimos Cuadrados Ordinarios.")

        coefs = model.params.reset_index()
        coefs.columns = ["Variable", "Coeficiente"]
        coefs["p-valor"] = model.pvalues.values
        interpretaciones = []
        for coef in coefs["Coeficiente"]:
            signo = "incrementa" if coef > 0 else "reduce"
            interpretaciones.append(f"A {signo} el valor esperado de Y.")
        coefs["Interpretación"] = interpretaciones

        st.markdown("#### Resultados del ajuste")
        st.dataframe(coefs)

    else:  # MNL
        st.info("Modelo MNL:")
        model = fit_mnl(df, FEATURES)
        probs = predict_mnl(model, df, FEATURES)
        st.markdown("#### Probabilidades predichas por categoría")
        st.dataframe(probs)

    # --- 3. Deliberación ---
    st.header("3. Deliberación epistémica")
    if "engine" not in st.session_state:
        st.session_state.engine = DeliberationEngine()
    prompt = st.text_input("Describe el análisis que quieres realizar:")
    if prompt:
        subqs = st.session_state.engine.generate_subquestions(prompt, FEATURES)
        st.subheader("Subpreguntas generadas")
        for i, q in enumerate(subqs, 1):
            ans = st.text_input(f"{i}. {q}")
            EpistemicNavigator.record(q, ans)
        st.markdown(f"**{len(subqs)}** subpreguntas generadas y registradas.")

    # --- 4. Validación ---
    st.header("4. Diagnóstico de modelo")
    diagnostics = check_model_diagnostics(df, model, FEATURES)
    st.json(diagnostics)

    # --- 5. Informe ---
    st.header("5. Informe final")
    if st.button("Generar informe PDF"):
        report = build_report(df, model, st.session_state.engine, diagnostics)
        export_pdf(report, "informe_simulador.pdf")
        st.success("Informe generado: informe_simulador.pdf")

if __name__ == "__main__":
    main()
