import numpy as np
from io import StringIO
from datetime import datetime

def build_report(df, model, engine, diagnostics):
    buf = StringIO()
    buf.write(f"Informe generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 1. Coeficientes del modelo
    buf.write("1. Coeficientes del modelo:\n")
    try:
        for name, coef in zip(model.params.index, model.params.values):
            # Si es escalar, imprime como número
            if np.isscalar(coef):
                buf.write(f"  - {name}: {coef:.4f}\n")
            else:
                # Si es array, imprime todos los valores
                buf.write(f"  - {name}: {np.array2string(np.asarray(coef), precision=4, separator=', ')}\n")
    except Exception as e:
        buf.write(f"  (Error al mostrar coeficientes: {e})\n")
    buf.write("\n")
    
    # 2. Diagnósticos
    buf.write("2. Diagnósticos del modelo:\n")
    try:
        for k, v in diagnostics.items():
            # Mostrar floats con 4 decimales, arrays como string
            if isinstance(v, float):
                buf.write(f"  - {k}: {v:.4f}\n")
            elif isinstance(v, np.ndarray):
                buf.write(f"  - {k}: {np.array2string(v, precision=4, separator=', ')}\n")
            else:
                buf.write(f"  - {k}: {v}\n")
    except Exception as e:
        buf.write(f"  (Error en diagnósticos: {e})\n")
    buf.write("\n")
    
    # 3. Trazabilidad deliberativa
    buf.write("3. Trazabilidad deliberativa (pasos):\n")
    try:
        tracker = engine.tracker.get_tracker() if hasattr(engine, "tracker") else {}
        steps = tracker.get("steps", [])
        if not steps and hasattr(engine, "get_tracker"):
            # Soporte alternativo
            steps = engine.get_tracker().get("steps", [])
        for i, step in enumerate(steps, 1):
            q = step.get('question', '')
            a = step.get('answer', '')
            buf.write(f"  {i}. {q}\n")
            buf.write(f"     Respuesta: {a}\n")
    except Exception as e:
        buf.write(f"  (Error en trazabilidad deliberativa: {e})\n")
    buf.write("\n")
    
    # 4. Resumen de datos
    buf.write("4. Resumen de datos (primeras filas):\n")
    try:
        buf.write(df.head().to_string())
        buf.write("\n")
    except Exception as e:
        buf.write(f"  (Error en resumen de datos: {e})\n")
    
    return buf.getvalue().encode()
