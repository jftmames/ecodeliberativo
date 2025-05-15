# epistemic_metrics.py

def compute_eee(tracker, max_steps=10):
    """
    Calcula métricas simples de Equilibrio Erotético Epistémico.
    Ejemplo: profundidad, número de subpreguntas, % contestadas, diversidad.
    """
    steps = tracker.get("steps", [])
    if not steps:
        return {"Profundidad": 0, "Subpreguntas": 0, "Contestadas": 0.0, "Diversidad": 0}
    
    total = len(steps)
    contestadas = sum(1 for s in steps if s.get("answer"))
    padres = set(s.get("parent") for s in steps)
    profundidad = max((0 if s.get("parent") is None else 1 for s in steps), default=0)
    diversidad = len(padres)
    return {
        "Profundidad": profundidad,
        "Subpreguntas": total,
        "Contestadas": contestadas / total if total else 0,
        "Diversidad": diversidad
    }
