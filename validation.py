# validation.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats

def check_model_diagnostics(df: pd.DataFrame, model, features: list[str]) -> dict:
    """
    Realiza diagnósticos para el modelo ajustado.
    
    - Para OLS:
      * VIF de cada variable explicativa.
      * Test de Breusch-Pagan (heterocedasticidad).
      * R-squared y RMSE.
    - Para Logit:
      * Pseudo R-squared (McFadden).
      * Test de razón de verosimilitud.
      * Estadístico de Hosmer-Lemeshow (aprox.).
    - Para MNL: vacía (por implementar).
    
    Devuelve un dict con diagnósticos.
    """
    results = {}
    model_name = type(model.model).__name__

    # Preparamos X y y comunes
    X = sm.add_constant(df[features], has_constant='add')
    y = df['Y']

    if 'OLS' in model_name:
        # R-squared y RMSE
        results['R-squared'] = model.rsquared
        results['RMSE'] = float(np.sqrt(model.mse_resid))

        # VIF
        vif_data = []
        for i, feat in enumerate(X.columns):
            if feat == 'const':
                continue
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({'variable': feat, 'VIF': vif})
        results['VIF'] = vif_data

        # Breusch-Pagan
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        results['Breusch-Pagan'] = {
            'LM_stat': bp_test[0],
            'LM_pvalue': bp_test[1],
            'F_stat': bp_test[2],
            'F_pvalue': bp_test[3]
        }

    elif 'Logit' in model_name:
        # Pseudo R2
        results['Pseudo R-squared (McFadden)'] = model.prsquared

        # Likelihood ratio test
        llr, pval, df_diff = model.llr, model.llr_pvalue, model.df_model
        results['Likelihood Ratio'] = {
            'LLR_stat': llr,
            'p-value': pval,
            'df_model': df_diff
        }

        # Hosmer-Lemeshow (aprox): agrupamos en 10 deciles
        data = X.copy()
        data['Y'] = y
        data['pred'] = model.predict(X)
        data['decile'] = pd.qcut(data['pred'], 10, labels=False)
        hl = []
        for dec in range(10):
            sub = data[data['decile'] == dec]
            obs = sub['Y'].sum()
            exp = sub['pred'].sum()
            n = len(sub)
            hl.append({
                'decile': int(dec),
                'observed': int(obs),
                'expected': float(exp),
                'n': int(n)
            })
        # Chi-cuadrado HL
        hl_stat = sum((d['observed'] - d['expected'])**2 / (d['expected'] * (1 - d['expected']/d['n']) + 1e-6) for d in hl)
        results['Hosmer-Lemeshow'] = {
            'table': hl,
            'chi2': hl_stat,
            'df': 8,
            'p-value': 1 - stats.chi2.cdf(hl_stat, 8)
        }

    else:
        # MNL u otros
        results['info'] = "Diagnósticos para MNL aún no implementados."

    return results
