# report_generator.py

import io
import pandas as pd
from typing import Any, Dict
from datetime import datetime

def build_report(
    df: pd.DataFrame,
    model: Any,
    engine: Any,
    diagnostics: Dict[str, Any]
) -> bytes:
    """
    Genera un informe de texto con los resultados del modelo y diagnósticos.
    """
    buf = io.StringIO()
    buf.write(f"Informe generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    buf.write("1. Coeficientes del modelo:\n")
    for name, coef in zip(model.params.index, model.params.values):
        buf.write(f"  - {name}: {coef:.4f}\n")

    buf.write("\n2. Diagnósticos:\n")
    for k, v in diagnostics.items():
        if isinstance(v, float):
            buf.write(f"  - {k}: {v:.4f}\n")
        elif isinstance(v, dict):
            buf.write(f"  - {k}:\n")
            for subk, subv in v.items():
                if isinstance(subv, float):
                    buf.write(f"      * {subk}: {subv:.4f}\n")
                else:
                    buf.write(f"      * {subk}: {subv}\n")
        elif hasattr(v, "tolist"):  # e.g. pandas Series or numpy array
            arr = v.tolist()
            buf.write(f"  - {k}: [{', '.join(f'{x:.4f}' if isinstance(x, (float, int)) else str(x) for x in arr[:5])} ...]\n")
        else:
            buf.write(f"  - {k}: {v}\n")

    buf.write("\n3. Comentarios adicionales:\n")
    buf.write("  - Aquí puedes añadir interpretaciones, gráficos y conclusiones.\n")

    return buf.getvalue().encode("utf-8")
