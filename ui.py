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

def generar_datos_ejemplo(tipo):
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "precio": np.random.uniform(1, 10, n),
        "ingreso": np.random.uniform(20, 100, n),
        "edad": np.random.randint(18, 70, n),
    })
    if tipo == "OLS":
        df["Y"] = 1.5 * df["precio"] + 0.2 * df["ingreso"] + np.random.normal(0, 2, n)
        explicacion = "Y es una variable continua (ideal para regresión OLS: ejemplo, salario, precio)."
        nests = None
    elif tipo == "Logit/Probit":
        logits = 1 / (1 + np.exp(-(-1 + 0.3*df["precio"] - 0.05*df["ingreso"])))
        df["Y"] = np.random.binomial(1, logits)
        explicacion = "Y es binaria (0/1): ideal para modelos Logit o Probit (por ejemplo, compra/no compra)."
        nests = None
    elif tipo == "MNL":
        logits1 = 1 / (1 + np.exp(-(-1 + 0.2*df["precio"] - 0.05*df["ingreso"])))
        logits2 = 1 / (1 + np.exp(-(0.5 - 0.1*df["precio"] + 0.07*df["edad"])))
        probs = np.column_stack([1 - logits1 - logits2, logits1, logits2])
        probs = np.clip(probs, 0, 1)
        probs = probs / probs.sum(axis=1, keepdims=True)
        df["Y"] = [np.random.choice([0, 1, 2], p=p) for p in probs]
        explicacion = "Y tiene 3 categorías (0, 1, 2): ideal para Logit Multinomial (MNL), por ejemplo, elección entre 3 productos."
        nests = None
    elif tipo == "Poisson":
        mu = np.exp(0.5 + 0.1*df["precio"] - 0.02*df["ingreso"])
        df["Y"] = np.random.poisson(mu)
        explicacion = "Y es una variable de conteo: ideal para Poisson (por ejemplo, número de compras, incidencias)."
        nests = None
    elif tipo == "Nested Logit":
        # Datos apilados
        n_personas = 100
        n_alternativas = 3
        data = []
        nests = {0: [0, 1], 1: [2]}  # Ejemplo: alternativas 0 y 1 en un nest, 2 en otro
        for i in range(n_personas):
            ingreso = np.random.uniform(20, 100)
            edad = np.random.randint(18, 70)
            precios = np.random.uniform(1, 10, n_alternativas)
            utilidades = -0.5*precios + 0.03*ingreso + 0.01*edad + np.random.normal(0,1, n_alternativas)
            exp_utilidades = np.exp(utilidades - np.max(utilidades))
            probs = exp_utilidades / exp_utilidades.sum()
            eleccion = np.random.choice(range(n_alternativas), p=probs)
            for j in range(n_alternativas):
                data.append({
                    "obs_id": i,
                    "alt_id": j,
                    "choice": int(j == eleccion),
                    "precio": precios[j],
                    "ingreso": ingreso,
                    "edad": edad,
                    "nest": 0 if j in [0,1] else 1
                })
        df = pd.DataFrame(data)
        explicacion = (
            "Y = 'choice' (0/1), datos apilados: cada fila es una alternativa disponible por persona. "
            "Columnas mínimas: 'obs_id', 'alt_id', 'choice', variables explicativas. "
            "Ejemplo ideal para Nested Logit (pylogit)."
        )
    else:
        explicacion = "Tipo no reconocido."
        nests = None
    return df, explicacion, nests

def reset_session_state():
    st.session_state.model = None
    st.session_state.FEATURES = []
    st.session_state.root_prompt = None
    st.session_state.subqs = []
    st.session_state.manual_subq_key = 0
    st.session_state.parent_node = None
    EpistemicNavigator.clear_tracker()

def main():
    # --- Inicialización robusta del estado ---
    for var, default in [
        ("model", None),
        ("FEATURES", []),
        ("engine", DeliberationEngine()),
        ("root_prompt", None),
        ("parent_node", None),
        ("manual_subq_key", 0),
        ("subqs", []),
        ("tipo_ejemplo_actual", None),
        ("df", None),
        ("nests", None),
    ]:
        if var not in st.session_state:
            st.session_state[var] = default

    st.set_page_config(page_title="Simulador Econométrico-Deliberativo", layout="wide")
    st.title("Simulador Econométrico-Deliberativo para Decisiones de Consumo")

    # --- Selector de tipo de modelo/datos de ejemplo ---
    st.markdown("## Elige el tipo de ejemplo que quieres probar")
    tipo_ejemplo = st.selectbox(
        "Selecciona tipo de modelo para precargar datos de ejemplo:",
        ["OLS", "Logit/Probit", "MNL", "Poisson", "Nested Logit"],
        index=1,
        help="Elige según el tipo de análisis que desees (puedes subir tus propios datos después)."
    )

    # Generar datos y resetear estado si cambia el tipo de ejemplo
    if (st.session_state.df is None) or (st.session_state.get("tipo_ejemplo_actual") != tipo_ejemplo):
        df, explicacion, nests = generar_datos_ejemplo(tipo_ejemplo)
        st.session_state.df = df
        st.session_state.nests = nests
        st.session_state.tipo_ejemplo_actual = tipo_ejemplo
        reset_session_state()
    else:
        df = st.session_state.df
        nests = st.session_state.nests
        explicaciones = {
            "OLS": "Y es una variable continua (ideal para regresión OLS: ejemplo, salario, precio).",
            "Logit/Probit": "Y es binaria (0/1): ideal para modelos Logit o Probit (por ejemplo, compra/no compra).",
            "MNL": "Y tiene 3 categorías (0, 1, 2): ideal para Logit Multinomial (MNL), por ejemplo, elección entre 3 productos.",
            "Poisson": "Y es una variable de conteo: ideal para Poisson (por ejemplo, número de compras, incidencias).",
            "Nested Logit": (
                "Y = 'choice' (0/1), datos apilados: cada fila es una alternativa disponible por persona. "
                "Columnas mínimas: 'obs_id', 'alt_id', 'choice', variables explicativas. "
                "Ejemplo ideal para Nested Logit (pylogit)."
            )
        }
        explicacion = explicaciones[tipo_ejemplo]

    # Explicación para el usuario
    st.info(f"**Ejemplo precargado:** {explicacion}")
    if tipo_ejemplo == "Nested Logit":
        st.warning(
            "⚠️ El modelo Nested Logit requiere datos apilados/stacked, es decir, cada fila representa una alternativa disponible para cada observación. "
            "Las columnas mínimas son: 'obs_id' (id de la observación), 'alt_id' (id de alternativa), 'choice' (0/1 si fue elegida), más las variables explicativas y 'nest'."
        )

    # --- CONTROL de subida de archivo y reset total ---
    uploaded = st.file_uploader("Sube un CSV con tus datos (incluye Y)", type="csv")
    if uploaded:
        st.session_state.df = pd.read_csv(uploaded)
        st.session_state.nests = None
        st.session_state.tipo_ejemplo_actual = None
        reset_session_state()
        st.success("Nuevo archivo cargado. Por favor, selecciona variables explicativas y estima un modelo.")
        st.rerun()
        return

    tabs = st.tabs([
        "1. Datos", "2. Econometría", "3. Deliberación", "4. Diagnóstico", "5. Informe"
    ])

    # --- 1. Datos ---
    with tabs[0]:
        st.header("1. Datos")
        df = st.session_state.df
        st.write(df.head())
        if tipo_ejemplo == "Nested Logit":
            # En datos apilados, elige variables aparte de obs_id, alt_id, choice y nest
            posibles = [c for c in df.columns if c not in ["obs_id", "alt_id", "choice", "nest"]]
        else:
            posibles = [c for c in df.columns if c != "Y"]
        FEATURES = st.multiselect(
            "Selecciona variables explicativas:",
            posibles,
            default=st.session_state.FEATURES
        )
        st.session_state.FEATURES = FEATURES
        st.sidebar.markdown(f"Paso 1: Datos {'✅' if FEATURES else '⬜'}")
        if not FEATURES:
            st.warning("Selecciona al menos una variable.")

    # --- 2. Econometría ---
    with tabs[1]:
        st.header("2. Econometría")
        FEATURES = st.session_state.FEATURES
        if not FEATURES:
            st.warning("Selecciona variables explicativas en la pestaña de datos.")
        else:
            y_unique = st.session_state.df["Y"].nunique() if "Y" in st.session_state.df.columns else 2
            allowed_models = ["OLS", "Logit", "Probit", "Poisson"]
            if y_unique >= 3 and tipo_ejemplo != "Nested Logit":
                allowed_models.append("MNL")
            if tipo_ejemplo == "Nested Logit" or (st.session_state.df is not None and "choice" in st.session_state.df.columns):
                allowed_models.append("Nested Logit")
            allowed_models.append("Tobit")

            model_name = st.selectbox(
                "Selecciona el modelo econométrico",
                allowed_models,
                key="modelo_econometrico"
            )
            if tipo_ejemplo == "Nested Logit" or model_name == "Nested Logit":
                X = df[FEATURES]
                y = df["choice"]
            else:
                X = df[FEATURES]
                y = df["Y"]
            if st.button(f"Estimar modelo {model_name}"):
                with st.spinner(f"Estimando modelo {model_name}..."):
                    if model_name == "MNL":
                        st.session_state.model = fit_mnl(df, FEATURES)
                    elif model_name == "Nested Logit":
                        # Nested Logit con pylogit (requiere datos apilados)
                        from econometrics import estimate_nested_logit_pylogit
                        st.session_state.model = estimate_nested_logit_pylogit(
                            df, st.session_state.nests, alt_id_col="alt_id", obs_id_col="obs_id", choice_col="choice"
                        )
                    else:
                        st.session_state.model = estimate_model(model_name, X, y)
                    st.success(f"Modelo {model_name} estimado correctamente.")
                    st.write("Resumen del modelo:")
                    # pylogit tiene su propio summary
                    if model_name == "Nested Logit":
                        st.text(st.session_state.model.summary())
                    else:
                        st.text(st.session_state.model.summary())
                st.rerun()
                return

            if st.session_state.model is not None:
                if model_name == "MNL":
                    st.markdown("#### Probabilidades (dataset completo)")
                    prob_df = predict_mnl(st.session_state.model, df, FEATURES)
                    st.dataframe(prob_df)
                    st.subheader("Probabilidades (gráfico)")
                    st.line_chart(prob_df)
                    st.subheader("Simulación interactiva MNL")
                    sim_vals = {}
                    for feat in FEATURES:
                        mi, ma = float(df[feat].min()), float(df[feat].max())
                        sim_vals[feat] = st.slider(f"{feat}", mi, ma, float(df[feat].median()), key=f"sim_mnl_{feat}")
                    df_new = pd.DataFrame([sim_vals])
                    sim_probs = predict_mnl(st.session_state.model, df_new, FEATURES)
                    st.dataframe(sim_probs)
                    st.bar_chart(sim_probs.T)
                elif model_name == "Nested Logit":
                    st.text(st.session_state.model.summary())
                else:
                    coefs = st.session_state.model.params.reset_index()
                    coefs.columns = ["Variable", "Coeficiente"]
                    coefs["p-valor"] = st.session_state.model.pvalues.values
                    coefs["Interpretación"] = [
                        "Incrementa" if c > 0 else "Reduce" for c in coefs["Coeficiente"]
                    ]
                    st.dataframe(coefs)
                    try:
                        elas_df = compute_elasticities(st.session_state.model, df, FEATURES)
                        st.subheader("Elasticidades")
                        st.table(elas_df)
                        if not elas_df["Elasticidad"].isnull().all():
                            st.subheader("Elasticidades (gráfico)")
                            chart_data = elas_df.set_index("Variable")["Elasticidad"]
                            st.bar_chart(chart_data)
                    except Exception:
                        st.info("Elasticidades no disponibles para este modelo.")

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
                        pred = st.session_state.model.predict(Xnew)[0]
                    except Exception:
                        pred = np.nan
                    if "Logit" in str(type(st.session_state.model)) or "Probit" in str(type(st.session_state.model)):
                        st.write(f"**Probabilidad estimada** = {pred:.3f}")
                    else:
                        st.write(f"**Y estimado** = {pred:.3f}")

        st.sidebar.markdown(f"Paso 2: Econometría {'✅' if st.session_state.model is not None else '⬜'}")

    # --- 3. Deliberación ---
    with tabs[2]:
        st.header("3. Deliberación epistémica")
        FEATURES = st.session_state.FEATURES
        if not FEATURES:
            st.warning("Debes seleccionar variables explicativas en la pestaña de datos para deliberar.")
        elif st.session_state.model is None:
            st.warning("Debes estimar un modelo primero para simulación, diagnóstico e informe.")
            if st.button("Estimar modelo automáticamente desde Deliberación"):
                df = st.session_state.df
                model_name = "Logit"
                import statsmodels.api as sm
                X = sm.add_constant(df[FEATURES], has_constant="add")
                y = df["Y"]
                from statsmodels.discrete.discrete_model import Logit
                st.session_state.model = Logit(y, X).fit(disp=False)
                st.success(f"Modelo {model_name} estimado automáticamente.")
                st.rerun()
                return
            st.stop()
        else:
            if st.session_state.root_prompt is None:
                prompt = st.text_input("Describe el análisis que quieres realizar:")
                if prompt:
                    st.session_state.root_prompt = prompt
                    EpistemicNavigator.add_step(prompt, parent=None)
                    subqs = st.session_state.engine.generate_subquestions(prompt, FEATURES)
                    st.session_state.subqs = subqs
            else:
                st.success(f"Pregunta raíz: {st.session_state.root_prompt}")
                steps = EpistemicNavigator.get_tracker().get("steps", [])
                subqs = st.session_state.subqs
                manual_subqs = [s["question"] for s in steps if s.get("parent") == 0 and s["question"] not in subqs]

                for i, q in enumerate(subqs + manual_subqs, 1):
                    ans = st.text_input(f"{i}. {q}", key=f"ans_{i}")
                    if ans:
                        EpistemicNavigator.record(q, ans, parent=0)

                st.markdown("**Añadir subpregunta manual:**")
                new_subq = st.text_input(
                    "Nueva subpregunta",
                    key=f"manual_subq_input_{st.session_state.manual_subq_key}",
                    placeholder="Introduce una subpregunta y pulsa Añadir",
                )

                add_manual = st.button("Añadir subpregunta manual")
                if add_manual:
                    if new_subq.strip():
                        EpistemicNavigator.add_step(new_subq.strip(), parent=0)
                        st.session_state.manual_subq_key += 1
                        st.success("Subpregunta añadida.")
                    else:
                        st.warning("La subpregunta no puede estar vacía.")

                if st.button("Limpiar razonamiento"):
                    EpistemicNavigator.clear_tracker()
                    st.session_state.root_prompt = None
                    st.session_state.subqs = []
                    st.session_state.manual_subq_key = 0
                    st.rerun()
                    return

                if steps:
                    st.subheader("Árbol deliberativo")
                    try:
                        import graphviz
                        dot = "digraph razonamiento {\n"
                        for idx, step in enumerate(steps):
                            label = step["question"][:30].
