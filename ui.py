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
    X = sm.add_constant(df[features], has_constant="add")
    y = df["Y"]
    return Logit(y, X).fit(disp=False)

def main():
    st.set_page_config(page_title="Simulador Econométrico-Deliberativo", layout="wide")
    st.title("Simulador Econométrico-Deliberativo para Decisiones de Consumo")

    role = st.sidebar.selectbox("Modo de uso", ["Docente", "Consultor"])
    tabs = st.tabs(["1. Datos", "2. Econometría", "3. Deliberación", "4. Diagnóstico", "5. Informe"])

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

            # Coeficientes
            coefs = model.params.reset_index()
            coefs.columns = ["Variable", "Coeficiente"]
            coefs["p-valor"] = model.pvalues.values
            coefs["Interpretación"] = ["Incrementa" if c > 0 else "Reduce" for c in coefs["Coeficiente"]]
            st.dataframe(coefs)

            # Elasticidades
            elas_df = compute_elasticities(model, df, FEATURES)
            st.subheader("Elasticidades")
            st.table(elas_df)

            # Gráfico de elasticidades
            st.subheader("Elasticidades (gráfico)")
            chart_data = elas_df.set_index("Variable")["Elasticidad"]
            st.bar_chart(chart_data)

            # Curva Probabilidad vs Precio
            if "precio" in FEATURES:
                st.subheader("Curva: Probabilidad vs Precio")
                precio_grid = np.linspace(df["precio"].min(), df["precio"].max(), 100)
                grid = pd.DataFrame({feat: df[feat].mean() for feat in FEATURES}, index=precio_grid)
                grid["precio"] = precio_grid
                Xg = sm.add_constant(grid[FEATURES], has_constant="add")
                probs = model.predict(Xg)
                dfp = pd.DataFrame({"P(Y=1)": probs}, index=precio_grid)
                st.line_chart(dfp)

            # Simulación interactiva
            st.subheader("Simulación interactiva")
            sim_vals = {}
            for feat in FEATURES:
                mi, ma = float(df[feat].min()), float(df[feat].max())
                sim_vals[feat] = st.slider(f"{feat}", mi, ma, float(df[feat].median()))
            Xnew = pd.DataFrame([sim_vals])
            Xnew = sm.add_constant(Xnew, has_constant="add")
            prob = model.predict(Xnew)[0]
            st.write(f"**P(Y=1)** = {prob:.3f}")

        elif model_type == "OLS":
            X = sm.add_constant(df[FEATURES], has_constant="add")
            y = df["Y"]
            model = OLS(y, X).fit()

            st.markdown("### Modelo OLS estimado")
            st.latex(f"Y = β₀ + {' + '.join(FEATURES)}")

            # 1) Tabla de coeficientes
            coefs = model.params.reset_index()
            coefs.columns = ["Variable", "Coeficiente"]
            coefs["p-valor"] = model.pvalues.values
            coefs["Interpretación"] = ["Incrementa" if c > 0 else "Reduce" for c in coefs["Coeficiente"]]
            st.dataframe(coefs)

            # 2) Gráfico de regresión sobre una variable seleccionada
            vi = st.selectbox("Variable para graficar ajuste", FEATURES)
            st.subheader(f"Ajuste: Y vs {vi}")
            xx = np.linspace(df[vi].min(), df[vi].max(), 100)
            grid = pd.DataFrame({feat: df[feat].mean() for feat in FEATURES}, index=xx)
            grid[vi] = xx
            Xg = sm.add_constant(grid[FEATURES], has_constant="add")
            yg = model.predict(Xg)
            plot_df = pd.DataFrame({vi: xx, "Y_pred": yg})
            st.line_chart(plot_df.set_index(vi)["Y_pred"])

            # 3) Simulación interactiva
            st.subheader("Simulación interactiva")
            sim_vals = {}
            for feat in FEATURES:
                mi, ma = float(df[feat].min()), float(df[feat].max())
                sim_vals[feat] = st.slider(f"{feat}", mi, ma, float(df[feat].median()))
            Xnew = pd.DataFrame([sim_vals])
            Xnew = sm.add_constant(Xnew, has_constant="add")
            yhat = model.predict(Xnew)[0]
            st.write(f"**Y estimado** = {yhat:.3f}")

            # 4) Elasticidades aproximadas
            st.subheader("Elasticidades aproximadas (β·x̄/Ȳ)")
            ybar = y.mean()
            elas = []
            for i, feat in enumerate(FEATURES, start=1):
                β = model.params.iloc[i]
                xbar = df[feat].mean()
                elas.append({"Variable": feat, "Elasticidad": β * xbar / ybar})
            st.table(pd.DataFrame(elas))

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
                mi, ma = float(df[feat].min()), float(df[feat].max())
                sim_vals[feat] = st.slider(f"{feat}", mi, ma, float(df[feat].median()))
            df_new = pd.DataFrame([sim_vals])
            sim_probs = predict_mnl(model, df_new, FEATURES)
            st.dataframe(sim_probs)

        st.sidebar.markdown(f"Paso 2: Econometría {'✅' if model else '⬜'}")

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
                st.success(f"{len(subqs)} subpreguntas registradas.")

        tracker = EpistemicNavigator.get_tracker()
        steps = tracker.get("steps", [])
        if steps:
            st.subheader("Métricas Epistémicas (EEE)")
            metrics = compute_eee(tracker, max_steps=10)
            eeedf = pd.DataFrame.from_dict(metrics, orient="index", columns=["Valor"])
            eeedf.index.name = "Dimensión"
            st.table(eeedf)
        else:
            st.info("Registra al menos una respuesta para ver las métricas EEE.")

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
            is_pdf = report_bytes[:4] == b"%PDF"
            filename = "informe_deliberativo.pdf" if is_pdf else "informe_deliberativo.txt"
            mime = "application/pdf" if is_pdf else "text/plain"
            st.download_button("📥 Descargar Informe", data=report_bytes, file_name=filename, mime=mime)
            st.success("Informe listo para descargar.")
        st.sidebar.markdown("Paso 5: Informe 📄")

if __name__ == "__main__":
    main()
