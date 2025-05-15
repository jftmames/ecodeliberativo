from jinja2 import Template
import datetime

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Informe Simulador Econométrico-Deliberativo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1 { color: #00416a; }
        h2 { color: #236fa1; }
        .section { margin-bottom: 30px; }
        table { border-collapse: collapse; width: 80%; margin-bottom: 20px; }
        th, td { border: 1px solid #bbbbbb; padding: 8px; text-align: left; }
        th { background-color: #e9ecef; }
        .q { color: #0a3356; }
        .a { color: #1b6612; font-style: italic; }
    </style>
</head>
<body>
    <h1>Informe Simulador Econométrico-Deliberativo</h1>
    <div class="section">
        <b>Fecha:</b> {{ fecha }}
        <br><b>Modo de uso:</b> {{ modo }}
        <br><b>Modelo econométrico:</b> {{ modelo }}
        <br><b>Variable dependiente:</b> {{ y_var }}
        <br><b>Variables independientes:</b> {{ x_vars|join(", ") }}
    </div>
    <div class="section">
        <h2>Resumen del modelo</h2>
        <pre>{{ resumen }}</pre>
    </div>
    <div class="section">
        <h2>Coeficientes</h2>
        <table>
            <tr>
                <th>Variable</th>
                <th>Coeficiente</th>
            </tr>
            {% for v, c in coef.items() %}
            <tr>
                <td>{{ v }}</td>
                <td>{{ "{:0.4f}".format(c) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    <div class="section">
        <h2>Diagnóstico automático</h2>
        <table>
        {% for k, v in diagnostico.items() %}
            <tr>
                <th>{{ k }}</th>
                <td>{{ v }}</td>
            </tr>
        {% endfor %}
        </table>
    </div>
    <div class="section">
        <h2>Razonamiento deliberativo</h2>
        <table>
            <tr>
                <th>Pregunta</th>
                <th>Respuesta del usuario</th>
            </tr>
            {% for p, r in deliberacion %}
            <tr>
                <td class="q">{{ p }}</td>
                <td class="a">{{ r if r else "Sin respuesta" }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    <div class="section">
        <h2>Índice de Equilibrio Erotético (EEE)</h2>
        <b>Valor EEE:</b> {{ eee if eee is not none else "No calculado" }} <br>
        <b>Perfil:</b> {{ eee_texto if eee_texto else "No disponible" }}
    </div>
    <div class="section">
        <h2>Interpretación del usuario</h2>
        <p>{{ feedback if feedback else "No se proporcionó interpretación." }}</p>
    </div>
</body>
</html>
"""

def generar_informe_html(modo, modelo, y_var, x_vars, resumen, coef, diagnostico, deliberacion, eee=None, eee_texto="", feedback=""):
    fecha = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")
    template = Template(HTML_TEMPLATE)
    html = template.render(
        fecha=fecha,
        modo=modo,
        modelo=modelo,
        y_var=y_var,
        x_vars=x_vars,
        resumen=resumen,
        coef=coef.to_dict() if hasattr(coef, "to_dict") else dict(coef),
        diagnostico=diagnostico,
        deliberacion=deliberacion,
        eee=eee,
        eee_texto=eee_texto,
        feedback=feedback
    )
    return html
