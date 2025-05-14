import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS

from mnl import fit_mnl, predict_mnl
from elasticities import compute_elasticities
from deliberation_engine import DeliberationEngine
from navigator import EpistemicNavigator
from validation import check_model_diagnostics
from report_generator import build_report  # ya no necesitamos export_pdf aquí

def load_example_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "precio": np.random.uniform(1, 10, n),
        "ingreso": np.random.uniform(20, 100, n),
        "edad": np.random.randint(18, 70, n),
    })
    X = np.column_stack([np.ones(n), df["precio"], df["ingreso"], df["edad"]])
    beta = np.array([-1.0, -0.5, 0.03, 0.01])
    logits = 1 / (1 + np.exp(-X.dot(beta)))
    df["Y"] = np.random.binomial(1, logits)
    return df

def fit_logit(df: pd.DataFrame, features: list[str]):
    X = sm.add_constant(df[features])
    y = df["Y"]
    model = Logit(y, X).fit(disp=False)
    return model

def main():
    st.set_page_config(page_title="Simulador Econométrico-Deliberativo", layout="wide")
    st.title("Simulador Econométrico-Deliberativo para Decisiones de Consumo")

    role = st.sidebar.selectbox("Modo de uso", ["Docente", "Consultor"])

    tabs = st.tabs([
        "1. Datos",
        "2. Econometría",
        "3. Deliberación",
        "4. Diagnóstico",
        "5. Informe"
    ])

    # --- 1. Datos ---
    with tabs[0]:
        st.header("1. Datos")
        uploaded = st.file_uploader("Sube un CSV con tus datos (incluye Y)", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            st.info("Usando datos de ejemplo.")
            df = load_example_data()
        st.write(df.head())
        FEATURES = st.multiselect("Selecciona variables explicativas:", [c for c in df.columns if c != "Y"])
        if not FEATURES:
            st.warning("Selecciona al menos una variable.")
        st.sidebar.markdown(f"Paso 1: Datos {'✅' if FEATURES else '⬜'}")

    if not FEATURES:
        return

    # --- 2. Econometría ---
    with tabs[1]:
        st.header("2. Econometría")
        model_type = st.radio("Elige modelo:", ["Logit", "OLS", "MNL"])
        model = None

        if model_type == "Logit":
            model = fit_logit(df, FEATURES)
            st.markdown("### Modelo Logit estimado")
            terms = " + ".join([f"β₍{i+1}₎·{FEATURES[i]}" for i in range(len(FEATURES))])
            st.latex(f"P(Y=1|X) = 1 / \\bigl(1 + e^{{-[β₀ + {terms}]}}\\bigr)")
            coefs = model.params.reset_index()
            coefs.columns = ["Variable", "Coeficiente"]
            coefs["p-valor"] = model.pvalues.values
            coefs["Interpretación"] = ["Incrementa" if c>0 else "Reduce" for c in coefs["Coeficiente"]]
            st.dataframe(coefs)
            st.subheader("Elasticidades")
            st.table(compute_elasticities(model, df, FEATURES))

            # Simulación interactiva
            st.subheader("Simulación de probabilidades")
            sim_vals = {}
            for feat in FEATURES:
                # asume numérico continuo
                min_, max_ = float(df[feat].min()), float(df[feat].max())
                sim_vals[feat] = st.slider(f"{feat}", min_, max_, float(df[feat].median()))
            # construimos exógenas
            Xnew = [1.0] + [sim_vals[feat] for feat in FEATURES]
            # siempre en formato 2D: lista de listas
            prob = model.predict([Xnew])[0]
            st.write(f"**P(Y=1)** para esos valores: {prob:.3f}")

        elif model_type == "OLS":
            X = sm.add_constant(df[FEATURES])
            y = df["Y"]
            model = OLS(y, X).fit()
            st.markdown("### Modelo OLS estimado")
            st.latex(f"Y = β₀ + {' + '.join(FEATURES)}")
            coefs = model.params.reset_index()
            coefs.columns = ["Variable", "Coeficiente"]
            coefs["p-valor"] = model.pvalues.values
            coefs["Interpretación"] = ["Incrementa" if c>0 else "Reduce" for c in coefs["Coeficiente"]]
            st.dataframe(coefs)

        else:  # MNL
            st.info("Modelo MNL seleccionado")
            model = fit_mnl(df, FEATURES)
            st.markdown("#### Probabilidades predichas (todo el dataset)")
            st.dataframe(predict_mnl(model, df, FEATURES))

            # Simulación interactiva
            st.subheader("Simulación de probabilidades MNL")
            sim_vals = {}
            for feat in FEATURES:
                min_, max_ = float(df[feat].min()), float(df[feat].max())
                sim_vals[feat] = st.slider(f"{feat}", min_, max_, float(df[feat].median()))
            # un solo caso de prueba
            df_new = pd.DataFrame([{feat: sim_vals[feat] for feat in FEATURES}])
            probs = predict_mnl(model, df_new, FEATURES)
            st.write("Probabilidades por alternativa:")
            st.dataframe(probs)

        st.sidebar.markdown(f"Paso 2: Econometría {'✅' if model else '⬜'}")

    if model is None:
        return

    # --- 3. Deliberación ---
    with tabs[2]:
        st.header("3. Deliberación epistémica")
        if 'engine' not in st.session_state:
            st.session_state.engine = DeliberationEngine()
        prompt = st.text_input("Describe el análisis que quieres realizar:")
        if prompt:
            subqs = st.session_state.engine.generate_subquestions(prompt, FEATURES)
            answers = []
            for i, q in enumerate(subqs, 1):
                a = st.text_input(f"{i}. {q}", key=f"ans_{i}")
                answers.append(a)
                EpistemicNavigator.record(q, a)
            if any(answers):
                st.success(f"{len(subqs)} subpreguntas registradas.")
        st.sidebar.markdown("Paso 3: Deliberación ⚙️")

    # --- 4. Diagnóstico ---
    with tabs[3]:
        st.header("4. Diagnóstico del modelo")
        diagnostics = check_model_diagnostics(df, model, FEATURES)
        st.json(diagnostics)
        st.sidebar.markdown("Paso 4: Diagnóstico ✅")

    # --- 5. Informe ---
    with tabs[4]:
        st.header("5. Informe final")
        if st.button("Generar informe"):
            report_bytes = build_report(df, model, st.session_state.engine, diagnostics)

            # Detectar si es PDF (encabezado %PDF) o texto plano
            is_pdf = report_bytes[:4] == b"%PDF"
            filename = "informe_deliberativo.pdf" if is_pdf else "informe_deliberativo.txt"
            mime = "application/pdf" if is_pdf else "text/plain"

            st.download_button(
                label="📥 Descargar Informe",
                data=report_bytes,
                file_name=filename,
                mime=mime
            )
            st.success("Informe listo para descargar.")
        st.sidebar.markdown("Paso 5: Informe 📄")

if __name__ == "__main__":
    main()
