from econometrics import estimate_model

# --- 2. Econometría ---
with tabs[1]:
    st.header("2. Econometría")
    model_name = st.selectbox(
        "Selecciona el modelo econométrico",
        ["OLS", "Logit", "Probit", "MNL", "Poisson"],
        index=0
    )
    model = None

    X = df[FEATURES]
    y = df["Y"]

    if st.button(f"Estimar modelo {model_name}"):
        with st.spinner(f"Estimando modelo {model_name}..."):
            model = estimate_model(model_name, X, y)
            st.success(f"Modelo {model_name} estimado correctamente.")
            st.write("Resumen del modelo:")
            st.text(model.summary())

    # Mostrar coeficientes y elasticidades si el modelo ya se ha estimado
    if model is not None:
        coefs = model.params.reset_index()
        coefs.columns = ["Variable", "Coeficiente"]
        coefs["p-valor"] = model.pvalues.values
        coefs["Interpretación"] = ["Incrementa" if c > 0 else "Reduce" for c in coefs["Coeficiente"]]
        st.dataframe(coefs)

        # Elasticidades (si aplica)
        try:
            elas_df = compute_elasticities(model, df, FEATURES)
            st.subheader("Elasticidades")
            st.table(elas_df)
            st.subheader("Elasticidades (gráfico)")
            chart_data = elas_df.set_index("Variable")["Elasticidad"]
            st.bar_chart(chart_data)
        except Exception as e:
            st.info("Elasticidades no disponibles para este modelo.")

    st.sidebar.markdown(f"Paso 2: Econometría {'✅' if model is not None else '⬜'}")
