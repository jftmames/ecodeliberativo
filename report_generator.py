# report_generator.py

import io
from typing import Any, Dict
from epistemic_metrics import compute_eee
from navigator import EpistemicNavigator


def build_report(
    df: Any,
    model: Any,
    engine: Any,
    diagnostics: Dict[str, Any],
) -> bytes:
    """
    Construye un informe en texto plano (UTF-8) que incluye:
      1. Resumen de los datos.
      2. Resumen del modelo (summary).
      3. Diagnósticos del modelo.
      4. Registro de deliberación (preguntas y respuestas).
      5. Métricas EEE derivadas del tracker epistémico.
    Devuelve el contenido listo para descargar.
    """
    buf = io.StringIO()

    # 1. Encabezado y datos
    buf.write("=== INFORME DELIBERATIVO ===\n\n")
    buf.write("1. RESUMEN DE DATOS\n")
    buf.write(f"- Observaciones: {df.shape[0]}\n")
    buf.write(f"- Variables (incluyendo Y): {', '.join(df.columns.tolist())}\n\n")

    # 2. Modelo
    buf.write("2. RESUMEN DEL MODELO ESTIMADO\n")
    try:
        summary_text = model.summary().as_text()
    except Exception:
        # En caso de modelos MNL u otros sin .summary()
        summary_text = str(model)
    buf.write(summary_text + "\n\n")

    # 3. Diagnósticos
    buf.write("3. DIAGNÓSTICOS DEL MODELO\n")
    for key, val in diagnostics.items():
        buf.write(f"- {key}: {val}\n")
    buf.write("\n")

    # 4. Deliberación epistémica
    buf.write("4. DELIBERACIÓN EPISTÉMICA\n")
    tracker = EpistemicNavigator.get_tracker()
    steps = tracker.get("steps", [])
    if steps:
        for i, step in enumerate(steps, start=1):
            q = step.get("question", "")
            a = step.get("answer", "")
            buf.write(f"{i}. Pregunta: {q}\n")
            buf.write(f"   Respuesta: {a}\n\n")
    else:
        buf.write("No hay pasos de deliberación registrados.\n\n")

    # 5. Métricas EEE
    buf.write("5. MÉTRICAS EPISTÉMICAS (EEE)\n")
    metrics = compute_eee(tracker, max_steps=10)
    for dim, val in metrics.items():
        buf.write(f"- {dim}: {val:.3f}\n")

    # Resultado final
    text = buf.getvalue()
    buf.close()
    return text.encode("utf-8")
