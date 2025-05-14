import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS

from mnl import fit_mnl, predict_mnl
from elasticities import compute_elasticities


def fit_logit_model(df: pd.DataFrame, features: list[str]):
    """
    Ajusta un modelo Logit y devuelve el objeto ajustado.
    """
    X = sm.add_constant(df[features])
    model = Logit(df["Y"], X).fit(disp=False)
    return model


def run_econometria(df: pd.DataFrame, features: list[str]):
    """
    Panel de Econometría: selección de modelo, estimación y visualizaciones.
    Guarda el modelo en session_state['model'] para uso posterior.
    """
    # Selección de tipo de modelo
    model_type = st.radio("Elige modelo:", ["Logit", "OLS", "MNL"], key="model_type")
    model = None

    if model_type == "Logit":
        model = fit_logit_model(df, features)
        st.markdown("### Modelo Logit estimado")

        # Mostrar ecuación
        terms = " + ".join([f"β₍{i+1}₎·{features[i]}" for i in range(len(features))])
        st.latex(f"P(Y=1|X) = 1 / \\bigl(1 + e^{{-[β₀ + {terms}]}}\\bigr)")

        # Tabla de coeficientes
        coefs = model.params.reset_index()
        coefs.columns = ["Variable", "Coeficiente"]
        coefs["p-valor"] = model.pvalues.values
        coefs["Interpretación"] = ["Incrementa" if c>0 else "Reduce" for c in coefs["Coeficiente"]]
        st.table(coefs)

        # Elasticidades
        elas_df = compute_elasticities(model, df, features)
        st.subheader("Elasticidades")
        st.table(elas_df)

        # Gráfico de elasticidades
        if "Variable" in elas_df.columns and "Elasticidad" in elas_df.columns:
            st.subheader("Elasticidades (gráfico)")
            chart_data = elas_df.set_index("Variable")["Elasticidad"]
            st.bar_chart(chart_data)

        # Curva: Probabilidad vs precio
        if "precio" in features:
            st.subheader("Curva: Probabilidad vs Precio")
            precio_grid = np.linspace(df["precio"].min(), df["precio"].max(), 100)
            # Crear DataFrame con valores promedio para otras variables
            avg_vals = {f: df[f].mean() for f in features}
            grid_df = pd.DataFrame({**avg_vals, "precio": precio_grid})
            Xgrid = sm.add_constant(grid_df[features], has_constant="add")
            prob_grid = model.predict(Xgrid)
            st.line_chart(pd.Series(prob_grid, index=precio_grid, name="P(Y=1)"))

        # Simulación interactiva
        st.subheader("Simulación interactiva: Probabilidad")
        sim_vals = {}
        for feat in features:
            mi, ma = float(df[feat].min()), float(df[feat].max())
            sim_vals[feat] = st.slider(f"{feat}", mi, ma, float(df[feat].median()), key=f"sim_{feat}")
        Xnew = pd.DataFrame({feat: [sim_vals[feat] for feat in features]}, columns=features)
        Xnew = sm.add_constant(Xnew, has_constant="add")
        prob = model.predict(Xnew)[0]
        st.write(f"**P(Y=1)** = {prob:.3f}")

    elif model_type == "OLS":
        st.markdown("### Modelo OLS estimado")
        X = sm.add_constant(df[features])
        y = df["Y"]
        model = OLS(y, X).fit()

        # Mostrar ecuación
        st.latex(f"Y = β₀ + {' + '.join(features)}")

        # Tabla de coeficientes
        coefs = model.params.reset_index()
        coefs.columns = ["Variable", "Coeficiente"]
        coefs["p-valor"] = model.pvalues.values
        coefs["Interpretación"] = ["Incrementa" if c>0 else "Reduce" for c in coefs["Coeficiente"]]
        st.table(coefs)

    else:
        st.info("Modelo MNL seleccionado")
        model = fit_mnl(df, features)
        st.markdown("#### Probabilidades predichas (dataset completo)")
        prob_df = predict_mnl(model, df, features)
        st.dataframe(prob_df)

        # Gráfico de probabilidades
        st.subheader("Probabilidades (gráfico)")
        st.line_chart(prob_df)

        # Simulación interactiva MNL
        st.subheader("Simulación interactiva MNL")
        sim_vals = {}
        for feat in features:
            mi, ma = float(df[feat].min()), float(df[feat].max())
            sim_vals[feat] = st.slider(f"{feat}", mi, ma, float(df[feat].median()), key=f"mnl_{feat}")
        df_new = pd.DataFrame([{feat: sim_vals[feat] for feat in features}])
        sim_probs = predict_mnl(model, df_new, features)
        st.table(sim_probs)

    # Guardar modelo en session_state para usar en otros módulos
    st.session_state['model'] = model
