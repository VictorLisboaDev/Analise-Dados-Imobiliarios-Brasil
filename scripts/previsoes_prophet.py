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

    # ============================================
# 3. TREINAR MODELOS
# ============================================
print("\n🔄 3. Treinando modelos Prophet...")

modelos = {}
previsoes = {}

for uf in ufs_previsao:
    print(f"\n   📊 Treinando Prophet para {uf}...")
    
    # Criar e treinar modelo
    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    modelo.fit(dados_prophet[uf])
    modelos[uf] = modelo
    
    # Gerar previsões para 2026 (12 meses)
    future = modelo.make_future_dataframe(periods=12, freq='MS')
    forecast = modelo.predict(future)
    previsoes[uf] = forecast
    
    print(f"   ✅ {uf} treinado com sucesso!")

# ============================================
# 4. VISUALIZAR PREVISÕES
# ============================================
print("\n📊 4. Visualizando previsões...")

# Criar figura com subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, uf in enumerate(ufs_previsao):
    ax = axes[idx]
    
    # Dados históricos
    dados_uf = dados_prophet[uf]
    ax.scatter(dados_uf['ds'], dados_uf['y'], alpha=0.3, color='gray', label='Histórico')
    
    # Previsões
    forecast = previsoes[uf]
    ax.plot(forecast['ds'], forecast['yhat'], color='blue', linewidth=2, label='Previsão')
    ax.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='blue', alpha=0.2, label='Intervalo de Confiança'
    )
    
    # Destacar 2026
    forecast_2026 = forecast[forecast['ds'] >= '2026-01-01']
    if not forecast_2026.empty:
        ax.axvspan('2026-01-01', '2026-12-01', alpha=0.1, color='green', label='2026')
    
    ax.set_title(f'{uf} - Previsão de Preços', fontweight='bold', fontsize=14)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Preço (R$)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Previsões de Preços para 2026 (Prophet)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/graficos/prophet_previsoes_2026.png', dpi=300, bbox_inches='tight')
plt.show()
