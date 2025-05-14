# report_generator.py

import io
import pandas as pd

def build_report(df, model, engine, diagnostics: dict) -> bytes:
    """
    Genera un informe de texto plano con:
      - Resumen de datos
      - Resumen del modelo
      - Diagnósticos
      - Pasos de deliberación
      - Métricas EEE
    Devuelve un bytes object (texto UTF-8).
    """
    buf = io.StringIO()
    buf.write("=== Informe Deliberativo ===\n\n")

    # Datos
    buf.write("1) Datos (primeras 5 filas):\n")
    buf.write(df.head().to_string(index=False))
    buf.write("\n\n")

    # Modelo
    buf.write("2) Resumen del Modelo:\n")
    buf.write(model.summary().as_text())
    buf.write("\n\n")

    # Diagnósticos
    buf.write("3) Diagnósticos:\n")
    for k, v in diagnostics.items():
        buf.write(f"  - {k}: {v:.4f}\n")
    buf.write("\n")

    # Deliberación
    buf.write("4) Pasos de deliberación:\n")
    tracker = engine.get_tracker() if hasattr(engine, 'get_tracker') else {}
    steps = tracker.get("steps", [])
    for i, step in enumerate(steps, 1):
        buf.write(f"  {i}. Q: {step['question']}\n")
        buf.write(f"     A: {step['answer']}\n")
    buf.write("\n")

    # Métricas EEE
    from epistemic_metrics import compute_eee
    metrics = compute_eee(tracker)
    buf.write("5) Métricas Epistémicas (EEE):\n")
    for k, v in metrics.items():
        buf.write(f"  - {k}: {v:.4f}\n")

    return buf.getvalue().encode("utf-8")
