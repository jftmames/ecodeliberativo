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
            coefs["Interpretación"] = ["Incrementa" if c>0 else "Reduce" for c in coefs["Coeficiente"]]
            st.dataframe(coefs)

            # 2) Gráfico de regresión sobre una variable seleccionada
            vi = st.selectbox("Variable para graficar ajuste", FEATURES)
            st.subheader(f"Ajuste: Y vs {vi}")
            xx = np.linspace(df[vi].min(), df[vi].max(), 100)
            # construir valores promedio para el resto
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

        ```
**Qué hace cada sección**  
1. **Tabla de coeficientes**: Igual que antes.  
2. **Gráfico de ajuste**: Permite elegir una variable y ver la línea de regresión proyectada, manteniendo el resto en su media.  
3. **Simulación interactiva**: Sliders para todas las variables y cálculo de Ŷ instantáneo.  
4. **“Elasticidades” aproximadas**: Un indicador β·x̄/Ȳ para comparar en términos porcentuales el efecto marginal medio de cada variable sobre Ȳ.

Con esto, el bloque OLS quedará tan completo y dinámico como tus secciones de Logit y MNL. Reinicia la app y prueba de nuevo el modo OLS: ahora verás gráfico, simulación e “elasticidades”.
