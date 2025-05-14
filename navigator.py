# Módulo para registrar y recuperar el tracker epistémico

_tracker = {"steps": []}

class EpistemicNavigator:
    @staticmethod
    def record(question: str, answer: str, metadata: dict = None):
        """
        Añade un paso al tracker con pregunta, respuesta y metadatos.
        metadata puede incluir 'reformulation': bool
        """
        _tracker["steps"].append({
            "question": question,
            "answer": answer,
            "metadata": metadata or {}
        })

    @staticmethod
    def get_tracker() -> dict:
        """Devuelve el tracker completo."""
        return _tracker
