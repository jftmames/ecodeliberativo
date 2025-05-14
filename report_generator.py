try:
    from fpdf import FPDF
    _HAS_FPDF = True
except ImportError:
    _HAS_FPDF = False

import json


def build_report(df, model, engine, diagnostics) -> bytes:
    """
    Construye un informe (PDF si fpdf está instalado, o texto plano en bytes).
    """
    if not _HAS_FPDF:
        # Fallback: generar informe en texto plano
        lines = []
        lines.append("Informe Simulador Econométrico-Deliberativo")
        lines.append("\n1. Estadísticas de Datos:")
        lines.extend(df.describe().to_string().split('\n'))
        lines.append("\n2. Coeficientes del Modelo:")
        try:
            params = model.params
            for var, coef in params.items():
                lines.append(f"{var}: {coef:.4f}")
        except Exception:
            lines.append("No disponible para este modelo.")
        lines.append("\n3. Log de Deliberación:")
        log = engine.get_log() if hasattr(engine, 'get_log') else []
        if log:
            for entry in log:
                q = entry.get('question', '')
                a = entry.get('answer', '')
                lines.append(f"Q: {q}")
                lines.append(f"A: {a}")
        else:
            lines.append("Sin registro de deliberación.")
        lines.append("\n4. Diagnósticos del Modelo:")
        diag_text = json.dumps(diagnostics, indent=2)
        lines.extend(diag_text.split('\n'))
        return "\n".join(lines).encode('utf-8')

    # Con fpdf disponible, generar PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Informe Simulador Econométrico-Deliberativo", ln=True, align='C')
    pdf.ln(5)
    # Datos
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "1. Estadísticas de Datos", ln=True)
    pdf.set_font("Arial", "", 10)
    for line in df.describe().to_string().split("\n"):
        pdf.cell(0, 5, line, ln=True)
    pdf.ln(5)
    # Coeficientes
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "2. Coeficientes del Modelo", ln=True)
    pdf.set_font("Arial", "", 10)
    try:
        for var, coef in model.params.items():
            pdf.cell(0, 5, f"{var}: {coef:.4f}", ln=True)
    except Exception:
        pdf.cell(0, 5, "No disponible para este modelo.", ln=True)
    pdf.ln(5)
    # Deliberación
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
    # Diagnósticos
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "4. Diagnósticos del Modelo", ln=True)
    pdf.set_font("Arial", "", 10)
    for line in json.dumps(diagnostics, indent=2).split("\n"):
        pdf.cell(0, 5, line, ln=True)
    return pdf.output(dest='S').encode('latin-1')


def export_pdf(report_bytes: bytes, filename: str) -> None:
    """
    Guarda informe (PDF o texto plano) en disco.
    """
    mode = 'wb'
    data = report_bytes
    # Si es texto plano, guardar en UTF-8
    if not _HAS_FPDF:
        with open(filename.replace('.pdf', '.txt'), mode) as f:
            f.write(data)
    else:
        with open(filename, mode) as f:
            f.write(data)
