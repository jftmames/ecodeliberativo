# ui.py

import streamlit as st
from elasticities import calculate_elasticity
from reasoning_tracker import ReasoningTracker

# --- Modelos de ejemplo ---
class DummyOLS:
    def predict(self, X):
        return [2*x['precio'] + 0.5*x['ingreso'] for x in X]

class DummyLogit:
    def predict_proba(self, X):
        import math
        logits = [0.3*x['precio'] + 0.7*x['ingreso'] - 2 for x in X]
        probs = [1 / (1 + math.exp(-z)) for z in logits]
        return [[1-p, p] for p in probs]

# Aquí puedes cargar/definir otros modelos Dummy para MNL, Probit, etc.

models = {
    "OLS": DummyOLS(),
    "Logit": DummyLogit(),
    # Agrega aquí tus modelos reales
}

st.title("Simulador Econométrico-Deliberativo")

if "tracker" not in st.session_state:
    st.session_state.tracker = ReasoningTracker()

model_type = st.selectbox("Selecciona modelo", list(models.keys()))
model = models[model_type]
variable = st.selectbox("Variable a simular", ["precio", "ingreso"])
base_precio = st.number_input("Precio base", value=10)
base_ingreso = st.number_input("Ingreso base", value=50)
base_vals = {"precio": base_precio, "ingreso": base_ingreso}
nuevo_valor = st.number_input(f"Nuevo valor para {variable}", value=base_vals[variable]+1)

if st.button("Calcular elasticidad deliberativa"):
    elasticidad = calculate_elasticity(
        model, model_type, variable, base_vals, nuevo_valor, tracker=st.session_state.tracker
    )
    st.success(f"Elasticidad: {elasticidad if not isinstance(elasticidad, list) else [round(e, 3) for e in elasticidad]}")
    comentario = st.text_area("¿Qué pregunta o reflexión surge de este resultado?")
    if comentario:
        st.session_state.tracker.log_step(
            question=f"Reflexión sobre elasticidad de {variable}",
            answer=str(elasticidad),
            step_type="comentario_usuario",
            context={"comentario": comentario}
        )

st.header("Trayectoria deliberativa registrada")
for i, step in enumerate(st.session_state.tracker.get_steps()):
    st.markdown(f"**{i+1}.** *{step['question']}*\n\n> {step['answer']}\n\n_{step['type']}_")
    if step["context"].get("comentario"):
        st.info(f"Comentario del usuario: {step['context']['comentario']}")

if st.button("Descargar informe JSON"):
    st.download_button(
        label="Descargar JSON",
        data=st.session_state.tracker.to_json(),
        file_name="trayectoria_deliberativa.json",
        mime="application/json"
    )
