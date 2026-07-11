"""
PROJETO IMOBILIÁRIO BRASIL - MODELO PROPHET
Previsão de Tendência de Preços para 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('../outputs/graficos', exist_ok=True)

print("="*80)
print("📈 PROJETO IMOBILIÁRIO BRASIL - PREVISÕES PROPHET")
print("="*80)

# ============================================
# 1. CARREGAR DADOS
# ============================================
print("\n📂 1. Carregando dados...")
df = pd.read_csv('../outputs/dados_tratados.csv')
df['data'] = pd.to_datetime(df['data'])

print(f"✅ Dados carregados: {len(df)} registros")

# ============================================
# 2. PREPARAR DADOS PARA PROPHET
# ============================================
print("\n🔧 2. Preparando dados...")

def preparar_prophet(uf, df):
    """Prepara dados no formato Prophet"""
    df_uf = df[df['uf'] == uf].copy()
    df_prophet = pd.DataFrame({
        'ds': df_uf['data'],
        'y': df_uf['preco_medio_imovel']
    })
    return df_prophet

# UFs para previsão
ufs_previsao = ['SC', 'CE', 'SP', 'RJ']

dados_prophet = {}
for uf in ufs_previsao:
    dados_prophet[uf] = preparar_prophet(uf, df)
    print(f"   {uf}: {len(dados_prophet[uf])} registros")