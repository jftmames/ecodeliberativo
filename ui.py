import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

from statsmodels.discrete.discrete_model import Logit, MNLogit
from statsmodels.regression.linear_model import OLS

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

    st.sidebar.markdown("### Modo de uso")
    role = st.sidebar.selectbox("", ["Docente", "Consultor"])
    tabs = st.tabs(["1. Datos", "2. Econometría", "3. Deliberación", "4. Diagnóstico", "5. Informe"])

    # 1. Datos
    with tabs[0]:
        st.header("1. Datos")
        uploaded = st.file_uploader("Sube un CSV con tus datos (incluye columna Y)", type="csv")
        if uploaded:
            df = pd.read_csv(uploaded)
        else:
            st.info("Usando datos de ejemplo.")
            df = load_example_data()
        st.write(df.head())
        FEATURES = st.multiselect("Selecciona variables explicativas:", [c for c in df.columns if c != "Y"])
        if not FEATURES:
            st.warning("Selecciona al menos una variable para continuar.")
        st.sidebar.markdown(f"**Paso 1:** Datos {'✅' if FEATURES else '⬜'}")

    if not FEATURES:
        return

    # 2. Econometría
    with tabs[1]:
        st.header("2. Econometría")
        model_type = st.radio("Elige modelo:", ["Logit", "OLS", "MNL"])
        model = None

        if model_type == "Logit":
            model = fit_logit(df, FEATURES)
            st.markdown("#### Modelo Logit estimado")
            terms = " + ".join([f"β₍{i+1}₎·{FEATURES[i]}" for i in range(len(FEATURES))])
            st.latex(f"P(Y=1|X) = 1 / \\bigl(1 + e^{{-[β₀ + {terms}]}}\\bigr)")

            # Coeficientes
            coefs = model.params.reset_index()
            coefs.columns = ["Variable", "Coeficiente"]
            coefs["p-valor"] = model.pvalues.values
            coefs["Interpretación"] = ["Incrementa" if c>0 else "Reduce" for c in coefs["Coeficiente"]]
            st.dataframe(coefs)

            # Elasticidades
            elas_df = compute_elasticities(model, df, FEATURES)
            st.subheader("Elasticidades")
            st.table(elas_df)

            # Gráfico de elasticidades
            if {"Variable","Elasticidad"}.issubset(elas_df.columns):
                st.subheader("Elasticidades (gráfico)")
                st.bar_chart(elas_df.set_index("Variable")["Elasticidad"])
            else:
                st.warning("No se pudo graficar elasticidades (formato inesperado).")

            # Curva Probabilidad vs Precio
            if "precio" in FEATURES:
                st.subheader("Curva: Probabilidad vs Precio")
                precio_grid = np.linspace(df["precio"].min(), df["precio"].max(), 100)
                df_grid = pd.DataFrame({feat: df[feat].mean() for feat in FEATURES}, index=precio_grid)
                df_grid["precio"] = precio_grid
                X_grid = sm.add_constant(df_grid[FEATURES], has_constant="add")
                prob_grid = model.predict(X_grid)
                st.line_chart(pd.DataFrame({"P(Y=1)": prob_grid}, index=precio_grid))

            # Simulación interactiva
            st.subheader("Simulación interactiva")
            sim_vals = {
                feat: st.slider(feat, float(df[feat].min()), float(df[feat].max()), float(df[feat].median()))
                for feat in FEATURES
            }
            Xnew = [1.0] + [sim_vals[feat] for feat in FEATURES]
            st.write(f"**P(Y=1)** = {model.predict([Xnew])[0]:.3f}")

        elif model_type == "OLS":
            X = sm.add_constant(df[FEATURES])
            y = df["Y"]
            model = OLS(y, X).fit()
            st.markdown("#### Modelo OLS estimado")
            st.latex(f"Y = β₀ + {' + '.join(FEATURES)}")
            coefs = model.params.reset_index()
            coefs.columns = ["Variable", "Coeficiente"]
            coefs["p-valor"] = model.pvalues.values
            coefs["Interpretación"] = ["Incrementa" if c>0 else "Reduce" for c in coefs["Coeficiente"]]
            st.dataframe(coefs)

        else:  # MNL
            X = sm.add_constant(df[FEATURES])
            y = df["Y"].astype(int)
            model = MNLogit(y, X).fit(disp=False)

            st.markdown("#### Probabilidades (dataset completo)")
            prob_df = model.predict(X)
            st.dataframe(prob_df)

            # Gráfico de probabilidades
            st.subheader("Probabilidades (gráfico)")
            st.line_chart(prob_df)

            # Simulación interactiva MNL
            st.subheader("Simulación interactiva MNL")
            sim_vals = {
                feat: st.slider(feat, float(df[feat].min()), float(df[feat].max()), float(df[feat].median()))
                for feat in FEATURES
            }
            df_new = pd.DataFrame([sim_vals])
            X_new = sm.add_constant(df_new[FEATURES], has_constant="add")
            sim_prob = model.predict(X_new)
            st.dataframe(sim_prob)

        st.sidebar.markdown(f"**Paso 2:** Econometría {'✅' if model else '⬜'}")

    if model is None:
        return

    # 3. Deliberación
    with tabs[2]:
        st.header("3. Deliberación epistémica")

        # Inicializamos motor
        if "engine" not in st.session_state:
            st.session_state.engine = DeliberationEngine()
        # También nuestro fallback local
        if "epi_steps" not in st.session_state:
            st.session_state.epi_steps = []

        prompt = st.text_input("Describe el análisis que quieres realizar:")
        if prompt:
            subqs = st.session_state.engine.generate_subquestions(prompt, FEATURES)
            for i, q in enumerate(subqs, 1):
                ans = st.text_input(f"{i}. {q}", key=f"ans_{i}")
                # Registramos en navigator
                EpistemicNavigator.record(q, ans)
                # Y también en nuestro fallback local
                st.session_state.epi_steps.append({
                    "question": q,
                    "answer": ans,
                    "metadata": {}
                })
            if subqs:
                st.success(f"{len(subqs)} subpreguntas registradas.")

        # Recuperamos el tracker
        tracker = {}
        if hasattr(EpistemicNavigator, "get_tracker"):
            tracker = EpistemicNavigator.get_tracker()
        elif hasattr(EpistemicNavigator, "get_steps"):
            tracker = {"steps": EpistemicNavigator.get_steps()}
        else:
            tracker = {"steps": st.session_state.epi_steps}

        steps = tracker.get("steps", [])
        if steps:
            st.subheader("Métricas Epistémicas (EEE)")
            metrics = compute_eee(tracker, max_steps=10)
            eeedf = pd.DataFrame.from_dict(metrics, orient="index", columns=["Valor"])
            eeedf.index.name = "Dimensión"
            st.table(eeedf)
        else:
            st.info("Responde alguna subpregunta para calcular el EEE.")

        st.sidebar.markdown("**Paso 3:** Deliberación ⚙️")

    # 4. Diagnóstico
    with tabs[3]:
        st.header("4. Diagnóstico del modelo")
        diagnostics = check_model_diagnostics(df, model, FEATURES)
        st.json(diagnostics)
        st.sidebar.markdown("**Paso 4:** Diagnóstico ✅")

    # 5. Informe
    with tabs[4]:
        st.header("5. Informe final")
        if st.button("Generar informe"):
            report_bytes = build_report(df, model, st.session_state.engine, diagnostics)
            is_pdf = report_bytes[:4] == b"%PDF"
            filename = "informe_deliberativo.pdf" if is_pdf else "informe_deliberativo.txt"
            mime = "application/pdf" if is_pdf else "text/plain"
            st.download_button("📥 Descargar Informe", report_bytes, filename, mime)
            st.success("Informe listo para descargar.")
        st.sidebar.markdown("**Paso 5:** Informe 📄")

if __name__ == "__main__":
    main()

