import streamlit as st
import pandas as pd
import numpy as np

from econometrics import estimate_model
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

def main():
    st.set_page_config(page_title="Simulador Econométrico-Deliberativo", layout="wide")
    st.title("Simulador Econométrico-Deliberativo para Decisiones de Consumo")

    role = st.sidebar.selectbox("Modo de uso", ["Docente", "Consultor"])
    tabs = st.tabs([
        "1. Datos", "2. Econometría", "3. Deliberación", "4. Diagnóstico", "5. Informe"
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
        st.sidebar.markdown(f"Paso 1: Datos {'✅' if FEATURES else '⬜'}")

    if not FEATURES:
        return

    # --- 2. Econometría ---
    with tabs[1]:
        st.header("2. Econometría")
        model_name = st.selectbox(
            "Selecciona el modelo econométrico",
            ["OLS", "Logit", "Probit", "MNL", "Poisson"]
        )
        model = None

        X = df[FEATURES]
        y = df["Y"]

        # MNL requiere tratamiento especial
        if model_name == "MNL":
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
            st.bar_chart(sim_probs.T)
        else:
            if st.button(f"Estimar modelo {model_name}"):
                with st.spinner(f"Estimando modelo {model_name}..."):
                    model = estimate_model(model_name, X, y)
                    st.success(f"Modelo {model_name} estimado correctamente.")
                    st.write("Resumen del modelo:")
                    st.text(model.summary())

            if model is not None:
                coefs = model.params.reset_index()
                coefs.columns = ["Variable", "Coeficiente"]
                coefs["p-valor"] = model.pvalues.values
                coefs["Interpretación"] = [
                    "Incrementa" if c > 0 else "Reduce" for c in coefs["Coeficiente"]
                ]
                st.dataframe(coefs)

                # Elasticidades (si aplica)
                try:
                    elas_df = compute_elasticities(model, df, FEATURES)
                    st.subheader("Elasticidades")
                    st.table(elas_df)
                    if not elas_df["Elasticidad"].isnull().all():
                        st.subheader("Elasticidades (gráfico)")
                        chart_data = elas_df.set_index("Variable")["Elasticidad"]
                        st.bar_chart(chart_data)
                except Exception:
                    st.info("Elasticidades no disponibles para este modelo.")

                # Simulación interactiva
                st.subheader("Simulación interactiva")
                sim_vals = {}
                for feat in FEATURES:
                    mi, ma = float(df[feat].min()), float(df[feat].max())
                    sim_vals[feat] = st.slider(
                        f"{feat}", mi, ma, float(df[feat].median()), key=f"sim_{feat}"
                    )
                Xnew = pd.DataFrame([sim_vals])
                import statsmodels.api as sm
                Xnew = sm.add_constant(Xnew, has_constant="add")
                try:
                    pred = model.predict(Xnew)[0]
                except Exception:
                    pred = np.nan
                if "Logit" in str(type(model)) or "Probit" in str(type(model)):
                    st.write(f"**Probabilidad estimada** = {pred:.3f}")
                else:
                    st.write(f"**Y estimado** = {pred:.3f}")

        st.sidebar.markdown(f"Paso 2: Econometría {'✅' if model is not None else '⬜'}")

    if "model" not in locals() or model is None:
        return

    # --- 3. Deliberación ---
    with tabs[2]:
        st.header("3. Deliberación epistémica")
        if "engine" not in st.session_state:
            st.session_state.engine = DeliberationEngine()
        if "root_prompt" not in st.session_state:
            st.session_state.root_prompt = None
        if "parent_node" not in st.session_state:
            st.session_state.parent_node = None

        if st.session_state.root_prompt is None:
            prompt = st.text_input("Describe el análisis que quieres realizar:")
            if prompt:
                st.session_state.root_prompt = prompt
                EpistemicNavigator.add_step(prompt, parent=None)
                subqs = st.session_state.engine.generate_subquestions(prompt, FEATURES)
                st.session_state.subqs = subqs
        else:
            st.success(f"Pregunta raíz: {st.session_state.root_prompt}")
            subqs = getattr(st.session_state, "subqs", [])
            for i, q in enumerate(subqs, 1):
                ans = st.text_input(f"{i}. {q}", key=f"ans_{i}")
                if ans:
                    EpistemicNavigator.record(q, ans, parent=0)
            if st.button("Añadir subpregunta manual"):
                new_subq = st.text_input("Nueva subpregunta:", key="manual_subq")
                if new_subq:
                    EpistemicNavigator.add_step(new_subq, parent=0)
            if st.button("Limpiar razonamiento"):
                EpistemicNavigator.clear_tracker()
                st.session_state.root_prompt = None
                st.session_state.subqs = []
                st.experimental_rerun()

        # Visualización árbol deliberativo
        tracker = EpistemicNavigator.get_tracker()
        steps = tracker.get("steps", [])
        if steps:
            st.subheader("Árbol deliberativo")
            try:
                import graphviz
                dot = "digraph razonamiento {\n"
                for idx, step in enumerate(steps):
                    label = step["question"][:30].replace('"', "'")
                    dot += f'{idx} [label="{label}"];\n'
                    if step.get("parent") is not None:
                        dot += f'{step["parent"]} -> {idx};\n'
                dot += "}"
                st.graphviz_chart(dot)
            except Exception:
                st.info("Visualización de árbol no disponible (instala graphviz en requirements.txt).")
            st.subheader("Métricas Epistémicas (EEE)")
            metrics = compute_eee(tracker, max_steps=10)
            eeedf = pd.DataFrame.from_dict(metrics, orient="index", columns=["Valor"])
            eeedf.index.name = "Dimensión"
            st.table(eeedf)
        else:
            st.info("Registra al menos una respuesta para ver el árbol y métricas.")

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
            st.download_button(
                "📥 Descargar Informe", data=report_bytes,
                file_name=filename, mime=mime
            )
            st.success("Informe listo para descargar.")
        st.sidebar.markdown("Paso 5: Informe 📄")

if __name__ == "__main__":
    main()
