# utils.py
import streamlit as st
from functools import wraps

def cache_data(func):
    """
    Decorador para cachear funciones de carga de datos en Streamlit.
    """
    return st.cache(func)

def format_dataframe(df):
    """
    Formatea el DataFrame si hace falta (placeholder).
    """
    return df
