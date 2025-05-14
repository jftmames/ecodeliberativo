# report_generator.py
from typing import Dict

def build_report(data_summary: str,
                 model_summaries: Dict[str, str],
                 reasoning_log: Dict) -> str:
    """
    Construye un string HTML con:
      - Resumen de datos
      - Resúmenes de modelo
      - Registro de razonamiento (JSON)
    """
    html = f"""
    <html>
    <head><title>Informe de Simulación</title></head>
    <body>
      <h1>Informe de Simulación Econométrica</h1>
      <h2>1. Resumen de Datos</h2>
      <pre>{data_summary}</pre>
      <h2>2. Resultados de Modelos</h2>
    """
    for name, summary in model_summaries.items():
        html += f"<h3>Modelo: {name}</h3><pre>{summary}</pre>"
    html += f"""
      <h2>3. Registro de Razonamiento</h2>
      <pre>{reasoning_log}</pre>
    </body>
    </html>
    """
    return html

def export_pdf(html: str, output_path: str = "informe.pdf") -> str:
    """
    Exporta el HTML a un fichero. (Placeholder)
    Por defecto guarda el HTML en un .txt para descargar.
    """
    # Si en el futuro instalas pdfkit o WeasyPrint, aquí podrías llamar a esas librerías.
    # Por ahora, como placeholder, guardamos el HTML dentro de un .txt renombrado:
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
