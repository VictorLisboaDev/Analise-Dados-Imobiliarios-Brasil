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

# ============================================
# 5. COMPONENTES DAS PREVISÕES
# ============================================
print("\n📊 5. Componentes das previsões...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, uf in enumerate(ufs_previsao):
    ax = axes[idx]
    modelos[uf].plot_components(previsoes[uf], ax=ax)
    ax.set_title(f'{uf} - Componentes da Série', fontweight='bold')

plt.suptitle('Componentes das Previsões (Tendência + Sazonalidade)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/graficos/prophet_componentes.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 6. RESUMO DAS PREVISÕES PARA 2026
# ============================================
print("\n" + "="*80)
print("📊 PREVISÕES PARA 2026")
print("="*80)

resultados_2026 = []

for uf in ufs_previsao:
    forecast = previsoes[uf]
    forecast_2026 = forecast[forecast['ds'] >= '2026-01-01']
    
    if not forecast_2026.empty:
        preco_2025 = dados_prophet[uf]['y'].iloc[-1]
        preco_2026_medio = forecast_2026['yhat'].mean()
        preco_2026_min = forecast_2026['yhat_lower'].mean()
        preco_2026_max = forecast_2026['yhat_upper'].mean()
        crescimento = ((preco_2026_medio / preco_2025) - 1) * 100
        
        print(f"\n🚀 {uf}:")
        print(f"   Preço Dez/2025: R${preco_2025:,.2f}")
        print(f"   Preço Médio 2026: R${preco_2026_medio:,.2f}")
        print(f"   Intervalo: R${preco_2026_min:,.2f} - R${preco_2026_max:,.2f}")
        print(f"   Crescimento: {crescimento:.1f}%")
        
        resultados_2026.append({
            'UF': uf,
            'Preco_2025': preco_2025,
            'Preco_2026_medio': preco_2026_medio,
            'Preco_2026_min': preco_2026_min,
            'Preco_2026_max': preco_2026_max,
            'Crescimento_%': crescimento
        })

# Salvar resultados
resultados_df = pd.DataFrame(resultados_2026)
resultados_df = resultados_df.sort_values('Crescimento_%', ascending=False)
resultados_df.to_csv('../outputs/modelos/previsoes_2026.csv', index=False)

print("\n📊 Resumo das Previsões:")
print(resultados_df.to_string(index=False))

# ============================================
# 7. RELATÓRIO FINAL
# ============================================
print("\n" + "="*80)
print("📊 RELATÓRIO FINAL - PREVISÕES PROPHET")
print("="*80)

print(f"""
✅ PREVISÕES PARA 2026 CONCLUÍDAS!

📈 DESTAQUES:

1️⃣ Santa Catarina (SC)
   • Preço médio 2026: R${resultados_df[resultados_df['UF']=='SC']['Preco_2026_medio'].values[0]:,.2f}
   • Crescimento: {resultados_df[resultados_df['UF']=='SC']['Crescimento_%'].values[0]:.1f}%

2️⃣ Ceará (CE) - Seu estado!
   • Preço médio 2026: R${resultados_df[resultados_df['UF']=='CE']['Preco_2026_medio'].values[0]:,.2f}
   • Crescimento: {resultados_df[resultados_df['UF']=='CE']['Crescimento_%'].values[0]:.1f}%

3️⃣ São Paulo (SP)
   • Preço médio 2026: R${resultados_df[resultados_df['UF']=='SP']['Preco_2026_medio'].values[0]:,.2f}
   • Crescimento: {resultados_df[resultados_df['UF']=='SP']['Crescimento_%'].values[0]:.1f}%

4️⃣ Rio de Janeiro (RJ)
   • Preço médio 2026: R${resultados_df[resultados_df['UF']=='RJ']['Preco_2026_medio'].values[0]:,.2f}
   • Crescimento: {resultados_df[resultados_df['UF']=='RJ']['Crescimento_%'].values[0]:.1f}%

📁 ARQUIVOS GERADOS:
   • outputs/graficos/prophet_previsoes_2026.png
   • outputs/graficos/prophet_componentes.png
   • outputs/modelos/previsoes_2026.csv

🚀 PRÓXIMOS PASSOS:
   • Integrar com dados reais do FipeZAP
   • Criar dashboard com Streamlit para visualizar previsões
   • Adicionar mais UFs ao modelo
""")

print("✅ MODELO PROPHET CONCLUÍDO!")