# report_generator.py
from typing import Dict
import pdfkit  # si más adelante decides usar una librería de PDF

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

def export_pdf(html: str, output_path: str = "informe.pdf"):
    """
    Exporta el HTML a PDF (requiere configuración de wkhtmltopdf o similar).
    Por ahora placeholder.
    """
    # pdfkit.from_string(html, output_path)
    with open(output_path, "w") as f:
        f.write(html)  # fallback: guardar HTML como .pdf.txt
    return output_path
