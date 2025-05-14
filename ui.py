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
from epistemic_metrics import compute_eee
from validation import check_model_diagnostics
from report_generator import build_report

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
    return Logit(y, X).fit(disp=False)

def main():
    st.set_page_config(page_title="Simulador Econométrico-Deliberativo", layout="wide")
    st.title("Simulador Econométrico-Deliberativo para Decisiones de Consumo")

    # Sidebar role and progress
    role = st.sidebar.selectbox("Modo de uso", ["Docente", "Consultor"])
    steps_status = {1: "⬜", 2: "⬜", 3: "⚙️", 4: "✅", 5: "📄"}

    tabs = st.tabs([
        "1. Datos", "2. Econometría", "3. Deliberación",
        "4. Diagnóstico", "5. Informe"
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
        FEATURES = st.multiselect(
            "Selecciona variables explicativas:", 
            [c for c in df.columns if c != "Y"]
        )
        if not FEATURES:
            st.warning("Selecciona al menos una variable.")
        steps_status[1] = "✅" if FEATURES else "⬜"
        st.sidebar.markdown(f"Paso 1: Datos {steps_status[1]}")

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
            terms = " + ".join(
                f"β₍{i+1}₎·{FEATURES[i]}" for i in range(len(FEATURES))
            )
            st.latex(f"P(Y=1|X) = 1 / \\bigl(1 + e^{{-[β₀ + {terms}]}}\\bigr)")

            # Tabla de coeficientes
            coefs = model.params.reset_index()
            coefs.columns = ["Variable", "Coeficiente"]
            coefs["p-valor"] = model.pvalues.values
            coefs["Interpretación"] = [
                "Incrementa" if c > 0 else "Reduce"
                for c in coefs["Coeficiente"]
            ]
            st.dataframe(coefs)

            # Elasticidades
            st.subheader("Elasticidades")
            elas_df = compute_elasticities(model, df, FEATURES)
            st.table(elas_df)

            # Gráfico de elasticidades
            st.subheader("Elasticidades (gráfico)")
            if "Variable" in elas_df.columns and "Elasticidad" in elas_df.columns:
                chart_data = elas_df.set_index("Variable")["Elasticidad"]
                st.bar_chart(chart_data)

            # Curva Probabilidad vs Precio
            if "precio" in FEATURES:
                st.subheader("Curva: Probabilidad vs Precio")
                grid = np.linspace(df["precio"].min(), df["precio"].max(), 100)
                df_grid = pd.DataFrame({
                    feat: df[feat].mean() for feat in FEATURES
                }, index=grid)
                df_grid["precio"] = grid
                Xg = sm.add_constant(df_grid[FEATURES], has_constant="add")
                pg = model.predict(Xg)
                st.line_chart(pd.DataFrame({"P(Y=1)": pg}, index=grid))

            # Simulación interactiva
            st.subheader("Simulación interactiva")
            sim_vals = {}
            for feat in FEATURES:
                lo, hi = float(df[feat].min()), float(df[feat].max())
                sim_vals[feat] = st.slider(feat, lo, hi, float(df[feat].median()))
            xnew = [1] + [sim_vals[f] for f in FEATURES]
            pnew = model.predict([xnew])[0]
            st.write(f"**P(Y=1)** = {pnew:.3f}")

        elif model_type == "OLS":
            X = sm.add_constant(df[FEATURES])
            y = df["Y"]
            model = OLS(y, X).fit()
            st.markdown("### Modelo OLS estimado")
            st.latex(f"Y = β₀ + {' + '.join(FEATURES)}")
            coefs = model.params.reset_index()
            coefs.columns = ["Variable", "Coeficiente"]
            coefs["p-valor"] = model.pvalues.values
            coefs["Interpretación"] = [
                "Incrementa" if c > 0 else "Reduce"
                for c in coefs["Coeficiente"]
            ]
            st.dataframe(coefs)

        else:  # MNL
            st.info("Modelo MNL seleccionado")
            model = fit_mnl(df, FEATURES)
            st.markdown("#### Probabilidades (dataset completo)")
            prob_df = predict_mnl(model, df, FEATURES)
            st.dataframe(prob_df)

            st.subheader("Probabilidades (gráfico)")
            st.line_chart(prob_df)

            st.subheader("Simulación interactiva MNL")
            sim_vals = {}
            for feat in FEATURES:
                lo, hi = float(df[feat].min()), float(df[feat].max())
                sim_vals[feat] = st.slider(feat, lo, hi, float(df[feat].median()))
            df_new = pd.DataFrame([sim_vals])
            simp = predict_mnl(model, df_new, FEATURES)
            st.dataframe(simp)

        steps_status[2] = "✅" if model else "⬜"
        st.sidebar.markdown(f"Paso 2: Econometría {steps_status[2]}")

    if model is None:
        return

    # --- 3. Deliberación ---
    with tabs[2]:
        st.header("3. Deliberación epistémica")
        if "engine" not in st.session_state:
            st.session_state.engine = DeliberationEngine()

        prompt = st.text_input("Describe el análisis que quieres realizar:")
        if prompt:
            subqs = st.session_state.engine.generate_subquestions(prompt, FEATURES)
            for i, q in enumerate(subqs, 1):
                ans = st.text_input(f"{i}. {q}", key=f"ans_{i}")
                EpistemicNavigator.record(q, ans)
            if subqs:
                st.success(f"{len(subqs)} subpreguntas generadas.")

        # Mostrar métricas EEE
        tracker = EpistemicNavigator.get_tracker()
        if tracker["steps"]:
            st.subheader("Métricas Epistémicas (EEE)")
            metrics = compute_eee(tracker, max_steps=10)
            df_eee = pd.DataFrame.from_dict(metrics, orient="index", columns=["Valor"])
            df_eee.index.name = "Dimensión"
            st.table(df_eee)
            steps_status[3] = "✅"
        else:
            st.info("Responde al menos una subpregunta para ver EEE.")

        st.sidebar.markdown(f"Paso 3: Deliberación {steps_status[3]}")

    # --- 4. Diagnóstico ---
    with tabs[3]:
        st.header("4. Diagnóstico del modelo")
        diagnostics = check_model_diagnostics(df, model, FEATURES)
        st.json(diagnostics)
        steps_status[4] = "✅"
        st.sidebar.markdown(f"Paso 4: Diagnóstico {steps_status[4]}")

    # --- 5. Informe ---
    with tabs[4]:
        st.header("5. Informe final")
        if st.button("Generar informe"):
            report = build_report(df, model, st.session_state.engine, diagnostics)
            is_pdf = report[:4] == b"%PDF"
            name = "informe.pdf" if is_pdf else "informe.txt"
            mtype = "application/pdf" if is_pdf else "text/plain"
            st.download_button("📥 Descargar Informe", report, name, mtype)
            steps_status[5] = "✅"
        st.sidebar.markdown(f"Paso 5: Informe {steps_status[5]}")

if __name__ == "__main__":
    main()
