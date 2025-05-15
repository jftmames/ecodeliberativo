from datetime import datetime
from navigator import EpistemicNavigator
from epistemic_metrics import compute_eee
import io

def build_report(df, model, deliberation_engine, diagnostics):
    buf = io.StringIO()
    # Encabezado y contexto deliberativo
    buf.write("="*45 + "\n")
    buf.write("INFORME DEL SIMULADOR ECONOMÉTRICO-DELIBERATIVO\n")
    buf.write("="*45 + "\n")
    buf.write(f"Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    buf.write("Esta simulación sigue el paradigma del Código Deliberativo:\n")
    buf.write("- Registra todas las preguntas, respuestas y trayectorias exploradas.\n")
    buf.write("- Calcula el Índice de Equilibrio Erotético (EEE) para medir la calidad de la deliberación.\n\n")

    # Coeficientes del modelo
    buf.write("1. Coeficientes del modelo:\n")
    try:
        for name, coef in zip(model.params.index, model.params.values):
            buf.write(f"   - {name}: {coef:.4f}\n")
    except Exception:
        buf.write("   (No disponible para este modelo)\n")

    # Diagnósticos
    buf.write("\n2. Diagnósticos del modelo:\n")
    for k, v in diagnostics.items():
        if isinstance(v, float):
            buf.write(f"   - {k}: {v:.4f}\n")
        else:
            buf.write(f"   - {k}: {v}\n")

    # Historial deliberativo
    buf.write("\n3. Trayectoria deliberativa y subpreguntas:\n")
    tracker = EpistemicNavigator.get_tracker()
    steps = tracker.get("steps", [])
    for idx, step in enumerate(steps):
        buf.write(f"   - {idx+1}. {step['question']}\n")
        buf.write(f"       Respuesta: {step.get('answer', '')}\n")
        if step.get("type"):
            buf.write(f"       Tipo: {step.get('type')}\n")
        if step.get("context") and step["context"].get("comentario"):
            buf.write(f"       Comentario: {step['context']['comentario']}\n")

    # Interpretación deliberativa de simulaciones/elasticidades
    buf.write("\n3.b. Interpretaciones deliberativas de simulaciones:\n")
    for step in steps:
        if "elasticidad" in step.get("type", "") or "simulacion" in step.get("type", ""):
            buf.write(f"   > Pregunta: {step['question']}\n")
            buf.write(f"     Elasticidad/Simulación: {step.get('answer','')}\n")
            comentario = step.get("context", {}).get("comentario", "")
            if comentario:
                buf.write(f"     Reflexión: {comentario}\n")

    # Opcional: Árbol secuencial ASCII de la deliberación
    buf.write("\n3.c. Secuencia de nodos deliberativos:\n")
    for idx, step in enumerate(steps):
        prefix = "└─" if idx == len(steps)-1 else "├─"
        buf.write(f"   {prefix} {step['question']}\n")

    # Métrica EEE y explicación
    metrics = compute_eee(tracker, max_steps=10)
    buf.write("\n4. Índice de Equilibrio Erotético (EEE):\n")
    for dim, val in metrics.items():
        buf.write(f"   - {dim}: {val:.2f}\n")
    buf.write(
        "\nEl EEE mide la pluralidad, profundidad, trazabilidad y reversibilidad de la deliberación. "
        "Un valor más alto indica un razonamiento más robusto y plural.\n"
    )

    # Resumen de logros deliberativos
    buf.write("\nResumen de logros deliberativos:\n")
    buf.write(f"- Subpreguntas generadas: {len(steps)}\n")
    buf.write(f"- Valor EEE: {metrics.get('EEE', 'N/A')}\n")
    buf.write("- Diversidad de perspectivas, trazabilidad y pluralidad quedan reflejadas en el historial y árbol deliberativo.\n")

    # Cierre y créditos
    buf.write("\n---\n")
    buf.write("Este informe ha sido generado por el Simulador Econométrico-Deliberativo siguiendo el enfoque del Código Deliberativo.\n")

    return buf.getvalue().encode("utf-8")
