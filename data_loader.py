# data_loader.py
import pandas as pd
import numpy as np

def generate_synthetic_data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    Genera un DataFrame con datos simulados de consumidores.
    Columnas: precio, ingreso, edad, eleccion (0/1).
    """
    np.random.seed(seed)
    df = pd.DataFrame({
        "precio": np.random.uniform(1, 10, n),
        "ingreso": np.random.uniform(1000, 5000, n),
        "edad": np.random.randint(18, 70, n),
        "eleccion": np.random.choice([0, 1], size=n)
    })
    return df

def load_csv(uploaded_file) -> pd.DataFrame:
    """
    Carga un CSV subido por el usuario (p.ej. en Streamlit).
    """
    return pd.read_csv(uploaded_file)
