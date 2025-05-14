import pandas as pd
from navigator import EpistemicNavigator
from epistemic_metrics import compute_eee
from io import BytesIO

def build_report(df, model, engine, diagnostics: dict) -> bytes:
    """
    Construye un informe en texto plano (o PDF si se prefiere).
    Aquí devolvemos texto simple.
    """
    tracker = EpistemicNavigator.get_tracker()
    eee_metrics = compute_eee(tracker)

    lines = []
    lines.append("Informe Deliberativo\n")
    lines.append("=== Diagnóstico del Modelo ===")
    for k, v in diagnostics.items():
        lines.append(f"- {k}: {v}")
    lines.append("\n=== Métricas Epistémicas (EEE) ===")
    for k, v in eee_metrics.items():
        lines.append(f"- {k}: {v:.3f}")
    lines.append("\n=== Pasos de la Deliberación ===")
    for i, step in enumerate(tracker.get("steps", []), 1):
        q = step["question"]
        a = step.get("answer", "")
        lines.append(f"{i}. Q: {q}")
        lines.append(f"   A: {a}")

    text = "\n".join(lines)
    return text.encode("utf-8")
