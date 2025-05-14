from fpdf import FPDF
import json


def build_report(df, model, engine, diagnostics) -> bytes:
    """
    Construye un informe en PDF con datos clave del análisis:
    - Estadísticas de datos
    - Resumen de coeficientes del modelo
    - Log de deliberación
    - Diagnósticos
    Retorna contenido PDF en bytes.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    # Título
    pdf.cell(0, 10, "Informe Simulador Econométrico-Deliberativo", ln=True, align='C')
    pdf.ln(5)
    # Sección: Datos
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "1. Estadísticas de Datos", ln=True)
    pdf.set_font("Arial", "", 10)
    desc = df.describe().to_string().split('\n')
    for line in desc:
        pdf.cell(0, 5, line, ln=True)
    pdf.ln(5)
    # Sección: Coeficientes del modelo
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "2. Coeficientes del Modelo", ln=True)
    pdf.set_font("Arial", "", 10)
    try:
        params = model.params
        for var, coef in params.items():
            pdf.cell(0, 5, f"{var}: {coef:.4f}", ln=True)
    except Exception:
        pdf.cell(0, 5, "No disponible para este modelo.", ln=True)
    pdf.ln(5)
    # Sección: Deliberación
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "3. Log de Deliberación", ln=True)
    pdf.set_font("Arial", "", 10)
    log = engine.get_log() if hasattr(engine, 'get_log') else []
    if log:
        for entry in log:
            q = entry.get('question', '')
            a = entry.get('answer', '')
            pdf.multi_cell(0, 5, f"Q: {q}\nA: {a}\n")
    else:
        pdf.cell(0, 5, "Sin registro de deliberación.", ln=True)
    pdf.ln(5)
    # Sección: Diagnósticos
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "4. Diagnósticos del Modelo", ln=True)
    pdf.set_font("Arial", "", 10)
    diag_text = json.dumps(diagnostics, indent=2)
    for line in diag_text.split('\n'):
        pdf.cell(0, 5, line, ln=True)
    # Retornar PDF bytes
    return pdf.output(dest='S').encode('latin-1')


def export_pdf(report_bytes: bytes, filename: str) -> None:
    """
    Guarda en disco el informe PDF dado su contenido en bytes.
    """
    with open(filename, 'wb') as f:
        f.write(report_bytes)
