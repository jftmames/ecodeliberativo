import io
import pandas as pd

def build_report(df, model, engine, diagnostics: dict) -> bytes:
    """
    Genera un informe en texto plano que incluye:
     - Resumen del dataset
     - Resumen del modelo (texto)
     - Diagnósticos
     - Pasos deliberativos y métricas EEE
    """
    buf = io.StringIO()

    # 1. Datos
    buf.write("=== Informe Simulador Econométrico-Deliberativo ===\n\n")
    buf.write("**Resumen de datos**\n")
    buf.write(df.describe().to_string())
    buf.write("\n\n")

    # 2. Modelo
    buf.write("**Resumen del modelo**\n")
    buf.write(model.summary().as_text())
    buf.write("\n\n")

    # 3. Diagnóstico
    buf.write("**Diagnósticos**\n")
    for k, v in diagnostics.items():
        buf.write(f"- {k}: {v}\n")
    buf.write("\n")

    # 4. Deliberación
    buf.write("**Pasos deliberativos**\n")
    tracker = engine  # en ui.py le pasamos st.session_state.engine
    steps = tracker.get_tracker().get("steps", [])
    for i, s in enumerate(steps, 1):
        buf.write(f"{i}. Q: {s['question']}\n   A: {s['answer']}\n")
    buf.write("\n")

    # 5. Métricas EEE
    from epistemic_metrics import compute_eee
    metrics = compute_eee(tracker.get_tracker(), max_steps=10)
    buf.write("**Métricas Epistémicas (EEE)**\n")
    for dim, val in metrics.items():
        buf.write(f"- {dim}: {val:.3f}\n")

    return buf.getvalue().encode("utf-8")
