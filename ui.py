import streamlit as st
from elasticities import calculate_elasticity
from reasoning_tracker import ReasoningTracker
from trajectory_visualization import build_deliberative_tree
from report_generator import generate_pdf_report
import tempfile

# ---- Modelos Dummy de ejemplo ----
class DummyOLS:
    def predict(self, X):
        # X: lista de dicts {'precio':..., 'ingreso':...}
        return [2*x['precio'] + 0.5*x['ingreso'] for x in X]

class DummyLogit:
    def predict_proba(self, X):
        import math
        logits = [0.3*x['precio'] + 0.7*x['ingreso'] - 2 for x in X]
        probs = [1 / (1 + math.exp(-z)) for z in logits]
        return [[1-p, p] for p in probs]

# Aquí puedes añadir modelos Dummy para MNL, Probit, etc., o cargar tus modelos reales.
models = {
    "OLS": DummyOLS(),
    "Logit": DummyLogit(),
    # "MNL": DummyMNL(),
    # "Probit": DummyProbit(),
    # "Tobit": DummyTobit(),
    # "Poisson": DummyPoisson(),
    # "NestedLogit": DummyNestedLogit(),
}

st.set_page_config(page_title="Simulador Econométrico-Deliberativo", layout="centered")
st.title("Simulador Econométrico-Deliberativo")
st.markdown("""
Esta herramienta te permite explorar **elasticidades y simulaciones** de modelos econométricos en modo deliberativo: 
cada simulación queda registrada como paso del razonamiento, y puedes justificar o comentar cada exploración.
""")

# ---- Estado y Reasoning Tracker ----
if "tracker" not in st.session_state:
    st.session_state.tracker = ReasoningTracker()

# ---- Selección de modelo y variables ----
st.sidebar.header("Configuración del modelo")
model_type = st.sidebar.selectbox("Selecciona modelo", list(models.keys()))
model = models[model_type]

variable = st.sidebar.selectbox("Variable a simular", ["precio", "ingreso"])
base_precio = st.sidebar.number_input("Precio base", value=10)
base_ingreso = st.sidebar.number_input("Ingreso base", value=50)
base_vals = {"precio": base_precio, "ingreso": base_ingreso}
nuevo_valor = st.sidebar.number_input(f"Nuevo valor para {variable}", value=base_vals[variable]+1)

st.write(f"**Modelo seleccionado:** {model_type}")
st.write(f"**Variable:** {variable}")
st.write(f"**Valores base:** Precio={base_precio}, Ingreso={base_ingreso}")
st.write(f"**Nuevo valor para {variable}:** {nuevo_valor}")

# ---- Simulación y registro deliberativo ----
if st.button("Calcular elasticidad deliberativa"):
    try:
        elasticidad = calculate_elasticity(
            model, model_type, variable, base_vals, nuevo_valor, tracker=st.session_state.tracker
        )
        # Muestra resultado
        resultado = elasticidad if not isinstance(elasticidad, list) else [round(e, 3) for e in elasticidad]
        st.success(f"Elasticidad: {resultado}")
        # Permite comentario deliberativo
        comentario = st.text_area("¿Qué pregunta o reflexión surge de este resultado?", key=f"comentario_{len(st.session_state.tracker.get_steps())}")
        if comentario:
            st.session_state.tracker.log_step(
                question=f"Reflexión sobre elasticidad de {variable}",
                answer=str(resultado),
                step_type="comentario_usuario",
                context={"comentario": comentario}
            )
            st.info("Comentario registrado deliberativamente.")
    except Exception as e:
        st.error(f"Error en el cálculo: {e}")

# ---- Visualización de pasos deliberativos ----
st.header("Trayectoria deliberativa registrada")
for i, step in enumerate(st.session_state.tracker.get_steps()):
    st.markdown(f"**{i+1}.** *{step['question']}*")
    st.write(f"> {step['answer']}")
    st.caption(f"_Tipo: {step['type']}_")
    if step["context"].get("comentario"):
        st.info(f"Comentario: {step['context']['comentario']}")

# ---- Visualización gráfica de la trayectoria ----
st.header("Visualización gráfica de la trayectoria deliberativa")
steps = st.session_state.tracker.get_steps()
if steps:
    dot = build_deliberative_tree(steps)
    st.graphviz_chart(dot.source)
else:
    st.info("Aún no hay pasos deliberativos registrados.")

# ---- Exportación de informes ----
st.subheader("Exportar informe deliberativo")
col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="Descargar JSON",
        data=st.session_state.tracker.to_json(),
        file_name="trayectoria_deliberativa.json",
        mime="application/json"
    )

with col2:
    if st.button("Descargar informe PDF"):
        if not steps:
            st.warning("No hay pasos registrados.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                generate_pdf_report(steps, tmpfile.name)
                with open(tmpfile.name, "rb") as f:
                    st.download_button(
                        label="Descargar PDF",
                        data=f,
                        file_name="informe_deliberativo.pdf",
                        mime="application/pdf"
                    )

# ---- Limpiar sesión ----
st.sidebar.markdown("---")
if st.sidebar.button("Limpiar trayectoria deliberativa"):
    st.session_state.tracker.clear()
    st.experimental_rerun()
