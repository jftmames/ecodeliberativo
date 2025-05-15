# reasoning_tracker.py

import json
from datetime import datetime

class ReasoningTracker:
    def __init__(self):
        self.steps = []

    def log_step(self, question, answer, step_type="simulacion", context=None):
        self.steps.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "type": step_type,
            "context": context or {}
        })

    def to_json(self, filepath=None):
        if filepath:
            with open(filepath, "w") as f:
                json.dump(self.steps, f, indent=2, ensure_ascii=False)
        return json.dumps(self.steps, indent=2, ensure_ascii=False)

    def get_steps(self, filter_type=None):
        if filter_type:
            return [s for s in self.steps if s["type"] == filter_type]
        return self.steps

    def clear(self):
        self.steps = []

    def export_for_report(self):
        # Para usar en el informe: devuelve los pasos con pregunta, respuesta, tipo y comentarios
        return [
            {
                "Pregunta": step["question"],
                "Respuesta": step["answer"],
                "Tipo": step["type"],
                "Comentario": step["context"].get("comentario", "")
            }
            for step in self.steps
        ]
