from jinja2 import Template
import datetime
import pandas as pd

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
            <thead>
                <tr>
                    {% for header in coef_table.headers %}
                    <th>{{ header }}</th>
                    {% endfor %}
                </tr>
            </thead>
            <tbody>
                {% for row in coef_table.rows %}
                <tr>
                    {% for cell in row %}
                    <td>{{ cell }}</td>
                    {% endfor %}
                </tr>
                {% endfor %}
            </tbody>
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
    
    # --- INICIO DE LA CORRECCIÓN ---
    # Preparamos una estructura de tabla para los coeficientes que funcione
    # tanto para Series (modelos simples) como para DataFrames (modelo MNL)
    coef_table = {
        "headers": [],
        "rows": []
    }
    
    # Convertir el objeto de coeficientes a DataFrame si es necesario
    if isinstance(coef, pd.Series):
        coef_df = coef.to_frame("Coeficiente")
    else:  # Si ya es un DataFrame
        coef_df = coef.copy()

    # Formatear todos los valores numéricos a 4 decimales
    for col in coef_df.columns:
        # Asegurarse de que la columna sea numérica antes de formatear
        if pd.api.types.is_numeric_dtype(coef_df[col]):
            coef_df[col] = coef_df[col].map('{:0.4f}'.format)
    
    # Preparar la estructura final para la plantilla
    coef_df.reset_index(inplace=True)
    coef_df.rename(columns={'index': 'Variable'}, inplace=True)
    
    coef_table["headers"] = coef_df.columns.tolist()
    coef_table["rows"] = coef_df.values.tolist()
    # --- FIN DE LA CORRECCIÓN ---

    template = Template(HTML_TEMPLATE)
    
    # Renderizamos la plantilla con la nueva estructura de tabla para coeficientes
    html = template.render(
        fecha=fecha,
        modo=modo,
        modelo=modelo,
        y_var=y_var,
        x_vars=x_vars,
        resumen=resumen,
        coef_table=coef_table, # Pasamos la nueva estructura a la plantilla
        diagnostico=diagnostico,
        deliberacion=deliberacion,
        eee=eee,
        eee_texto=eee_texto,
        feedback=feedback
    )
    return html
