# report_generator.py

from fpdf import FPDF

class DeliberativePDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Informe Deliberativo - Simulador Econométrico", ln=True, align="C")
        self.ln(6)

    def add_step(self, idx, step):
        # Número y pregunta
        self.set_font("Arial", "B", 12)
        self.cell(0, 8, f"Paso {idx+1}", ln=True)
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 6, f"Pregunta: {step.get('question', '')}")
        # Respuesta
        self.set_font("Arial", "I", 11)
        self.multi_cell(0, 6, f"Respuesta: {step.get('answer', '')}")
        # Tipo de paso
        self.set_font("Arial", "", 10)
        self.cell(0, 6, f"Tipo: {step.get('type', '')}", ln=True)
        # Comentario deliberativo (si existe)
        comentario = step.get("context", {}).get("comentario", "")
        if comentario:
            self.set_text_color(30, 60, 110)
            self.set_font("Arial", "I", 10)
            self.multi_cell(0, 6, f"Comentario: {comentario}")
            self.set_text_color(0, 0, 0)
        self.ln(1)
        # Línea divisoria
        self.set_draw_color(180, 180, 180)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(130, 130, 130)
        self.cell(0, 10, f"Página {self.page_no()}", 0, 0, 'C')


def generate_pdf_report(steps, output_path):
    pdf = DeliberativePDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, 
        "Este informe recoge todas las simulaciones, preguntas y reflexiones realizadas en el Simulador Econométrico-Deliberativo bajo el enfoque del Código Deliberativo.\n"
        "Cada paso está registrado con transparencia y apertura epistémica.\n"
        " "
    )

    if not steps:
        pdf.set_font("Arial", "I", 12)
        pdf.cell(0, 10, "No hay pasos deliberativos registrados.", ln=True)
    else:
        for idx, step in enumerate(steps):
            pdf.add_step(idx, step)

    pdf.ln(6)
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 6, 
        "Informe generado automáticamente.\n"
        "Para trazabilidad o auditoría completa, consulte también el historial JSON disponible."
    )
    pdf.set_text_color(0, 0, 0)

    pdf.output(output_path)
    return output_path

