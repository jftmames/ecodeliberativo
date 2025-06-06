# ai_reasoning_generator.py

import openai
import json
import streamlit as st

def build_prompt(model_results: dict, user_profile: str, y_var: str, x_vars: list) -> str:
    """
    Construye un prompt detallado para enviar al modelo de OpenAI.
    """
    # Extraemos la información relevante de los resultados del modelo
    model_type = model_results.get('modelo', 'desconocido')
    summary_text = model_results.get('summary', 'No disponible.')
    diagnostics_dict = model_results.get('diagnostics', {})

    # Creamos un texto legible para los diagnósticos
    diagnostics_text = "\n".join([f"- {key}: {value}" for key, value in diagnostics_dict.items()])

    prompt = f"""
    Eres un economista experto y un coach de pensamiento crítico. Tu tarea es analizar los siguientes
    resultados de un modelo econométrico y generar 3-4 preguntas de reflexión que sean perspicaces y abiertas.
    Estas preguntas deben incitar al usuario a pensar críticamente sobre la validez del modelo, su interpretación
    y sus posibles implicaciones prácticas.

    El usuario que analizará las preguntas tiene un perfil de '{user_profile}'. Adapta las preguntas para que
    sean relevantes y desafiantes para este tipo de audiencia.

    Aquí están los detalles del análisis:
    - Tipo de Modelo: {model_type}
    - Variable Dependiente: {y_var}
    - Variables Independientes: {', '.join(x_vars)}

    ---
    RESUMEN DEL MODELO:
    {summary_text}
    ---
    DIAGNÓSTICOS DEL MODELO:
    {diagnostics_text}
    ---

    Basándote en esta información, genera entre 3 y 4 preguntas críticas. Las preguntas deben ir más allá
    de simples definiciones y explorar temas como:
    - Posibles sesgos por variables omitidas o endogeneidad.
    - La significancia práctica vs. la significancia estadística de los coeficientes.
    - La robustez de los supuestos del modelo (ej. linealidad, normalidad de errores, etc.).
    - Las limitaciones del análisis y qué pasos adicionales se podrían tomar.

    Devuelve las preguntas únicamente en formato de una lista JSON de strings. Por ejemplo:
    ["¿Qué posible variable omitida podría estar afectando tanto a 'X' como a 'Y'?", "¿Cómo justificarías la elección de este modelo frente a una alternativa no lineal?"]
    """
    return prompt

@st.cache_data(show_spinner=False) # Cache para no gastar tokens en la misma consulta
def generate_deliberative_questions(api_key: str, model_results: dict, user_profile: str, y_var: str, x_vars: list) -> list:
    """
    Llama a la API de OpenAI para generar las preguntas deliberativas.
    """
    if not api_key:
        return ["Error: La clave de API de OpenAI no ha sido proporcionada."]

    try:
        client = openai.OpenAI(api_key=api_key)
        prompt = build_prompt(model_results, user_profile, y_var, x_vars)

        response = client.chat.completions.create(
            model="gpt-4-turbo", # Puedes cambiar el modelo si lo deseas
            messages=[
                {"role": "system", "content": "Eres un economista experto."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
        )

        # Extraemos y parseamos la respuesta JSON
        content = response.choices[0].message.content
        questions = json.loads(content)
        
        if isinstance(questions, list):
            return questions
        else:
            return ["Error: La IA no devolvió una lista de preguntas en el formato esperado."]

    except json.JSONDecodeError:
        return ["Error: No se pudo decodificar la respuesta de la IA. La respuesta fue: " + content]
    except Exception as e:
        return [f"Error al contactar con la API de OpenAI: {e}"]
