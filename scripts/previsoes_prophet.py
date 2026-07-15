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
os.makedirs('outputs/graficos', exist_ok=True)
os.makedirs('outputs/modelos', exist_ok=True)

print("="*80)
print("📈 PROJETO IMOBILIÁRIO BRASIL - PREVISÕES PROPHET")
print("="*80)

# ============================================
# 1. CARREGAR DADOS
# ============================================
print("\n📂 1. Carregando dados...")

try:
    df = pd.read_csv('outputs/dados_tratados.csv')
    df['data'] = pd.to_datetime(df['data'])
    print(f"✅ Dados carregados: {len(df)} registros")
    print(f"📅 Período: {df['data'].min()} a {df['data'].max()}")
    print(f"📍 UFs disponíveis: {df['uf'].unique().tolist()}")
except FileNotFoundError:
    print("❌ Arquivo outputs/dados_tratados.csv não encontrado!")
    print("   Execute primeiro: python 04_modelos_preditivos.py")
    exit()

# ============================================
# 2. PREPARAR DADOS PARA PROPHET
# ============================================
print("\n🔧 2. Preparando dados...")

def preparar_prophet(uf, df):
    """Prepara dados no formato Prophet"""
    df_uf = df[df['uf'] == uf].copy()
    if df_uf.empty:
        print(f"   ⚠️ UF {uf} não encontrada nos dados!")
        return None
    
    # Verificar se há dados suficientes
    if len(df_uf) < 12:
        print(f"   ⚠️ UF {uf} tem apenas {len(df_uf)} registros (mínimo 12)")
        return None
    
    # Verificar se há dados completos
    if df_uf['preco_medio_imovel'].isnull().any():
        print(f"   ⚠️ UF {uf} tem dados faltantes!")
        return None
    
    df_prophet = pd.DataFrame({
        'ds': df_uf['data'],
        'y': df_uf['preco_medio_imovel']
    })
    
    return df_prophet

# UFs para previsão (apenas as que existem nos dados)
ufs_disponiveis = df['uf'].unique().tolist()
ufs_prioridade = ['SC', 'CE', 'SP', 'RJ', 'RS', 'MG', 'DF', 'GO', 'BA', 'AM']

ufs_previsao = []
for uf in ufs_prioridade:
    if uf in ufs_disponiveis:
        ufs_previsao.append(uf)

print(f"   UFs selecionadas para previsão: {ufs_previsao}")

dados_prophet = {}
for uf in ufs_previsao:
    dados = preparar_prophet(uf, df)
    if dados is not None:
        dados_prophet[uf] = dados
        print(f"   ✅ {uf}: {len(dados)} registros preparados")

# Verificar se há dados
if not dados_prophet:
    print("\n❌ Nenhum dado disponível para previsão!")
    print("   Verifique se o arquivo dados_tratados.csv existe e contém dados.")
    exit()

# ============================================
# 3. TREINAR MODELOS
# ============================================
print("\n🔄 3. Treinando modelos Prophet...")

modelos = {}
previsoes = {}

for uf, dados in dados_prophet.items():
    print(f"\n   📊 Treinando Prophet para {uf}...")
    
    try:
        # Criar e treinar modelo
        modelo = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            interval_width=0.95
        )
        modelo.fit(dados)
        modelos[uf] = modelo
        
        # Gerar previsões para 2026 (12 meses)
        future = modelo.make_future_dataframe(periods=12, freq='MS')
        forecast = modelo.predict(future)
        previsoes[uf] = forecast
        
        print(f"   ✅ {uf} treinado com sucesso! Previsões para 2026 geradas.")
        
    except Exception as e:
        print(f"   ❌ Erro ao treinar {uf}: {str(e)}")

# Verificar se há previsões
if not previsoes:
    print("\n❌ Nenhuma previsão foi gerada!")
    exit()

# ============================================
# 4. VISUALIZAR PREVISÕES
# ============================================
print("\n📊 4. Visualizando previsões...")

# Criar figura com subplots (ajustado para o número de UFs)
n_plots = len(ufs_previsao)
n_cols = 2
n_rows = (n_plots + 1) // 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
# Garantir que axes seja uma lista, mesmo com 1 plot
if n_plots == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, uf in enumerate(ufs_previsao):
    if idx >= len(axes):
        break
        
    ax = axes[idx]
    
    # Dados históricos
    dados_uf = dados_prophet[uf]
    ax.scatter(dados_uf['ds'], dados_uf['y'], alpha=0.3, color='gray', 
               label='Histórico', s=20)
    
    # Previsões
    forecast = previsoes[uf]
    ax.plot(forecast['ds'], forecast['yhat'], color='blue', linewidth=2, 
            label='Previsão')
    ax.fill_between(
        forecast['ds'],
        forecast['yhat_lower'],
        forecast['yhat_upper'],
        color='blue', alpha=0.2, label='Intervalo de Confiança (95%)'
    )
    
    # Destacar 2026
    forecast_2026 = forecast[forecast['ds'] >= '2026-01-01']
    if not forecast_2026.empty:
        ax.axvspan('2026-01-01', '2026-12-01', alpha=0.1, color='green', 
                   label='2026', zorder=0)
    
    # Adicionar linha do último valor
    ultimo_valor = dados_uf['y'].iloc[-1]
    ax.axhline(y=ultimo_valor, color='red', linestyle='--', alpha=0.5, 
               label=f'Último preço: R${ultimo_valor:,.0f}')
    
    ax.set_title(f'{uf} - Previsão de Preços', fontweight='bold', fontsize=14)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Preço (R$)', fontsize=12)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# Remover subplots vazios
for idx in range(len(ufs_previsao), len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Previsões de Preços para 2026 (Prophet)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/graficos/prophet_previsoes_2026.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Gráfico salvo: outputs/graficos/prophet_previsoes_2026.png")

# ============================================
# 5. COMPONENTES DAS PREVISÕES (Simplificado)
# ============================================
print("\n📊 5. Componentes das previsões...")

# Para cada UF, mostrar componentes separadamente
for uf in ufs_previsao:
    try:
        print(f"\n   📊 Componentes para {uf}:")
        modelo = modelos[uf]
        forecast = previsoes[uf]
        
        # Plotar componentes (Prophet já mostra)
        fig_comp = modelo.plot_components(forecast)
        plt.suptitle(f'{uf} - Componentes da Série', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'../outputs/graficos/prophet_componentes_{uf}.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(f"   ✅ Componentes salvos: prophet_componentes_{uf}.png")
        
    except Exception as e:
        print(f"   ⚠️ Erro ao plotar componentes para {uf}: {str(e)}")

# ============================================
# 6. RESUMO DAS PREVISÕES PARA 2026
# ============================================
print("\n" + "="*80)
print("📊 PREVISÕES PARA 2026")
print("="*80)

resultados_2026 = []

for uf, forecast in previsoes.items():
    try:
        # Previsões para 2026
        forecast_2026 = forecast[forecast['ds'] >= '2026-01-01']
        
        if not forecast_2026.empty:
            # Último valor histórico (2025)
            dados_uf = dados_prophet[uf]
            preco_2025 = dados_uf['y'].iloc[-1]
            
            # Médias para 2026
            preco_2026_medio = forecast_2026['yhat'].mean()
            preco_2026_min = forecast_2026['yhat_lower'].mean()
            preco_2026_max = forecast_2026['yhat_upper'].mean()
            
            # Crescimento
            crescimento = ((preco_2026_medio / preco_2025) - 1) * 100
            
            print(f"\n🚀 {uf}:")
            print(f"   Preço Dez/2025: R${preco_2025:,.2f}")
            print(f"   Preço Médio 2026: R${preco_2026_medio:,.2f}")
            print(f"   Intervalo (95%): R${preco_2026_min:,.2f} - R${preco_2026_max:,.2f}")
            print(f"   Crescimento: {crescimento:.1f}%")
            
            resultados_2026.append({
                'UF': uf,
                'Preco_2025': round(preco_2025, 2),
                'Preco_2026_medio': round(preco_2026_medio, 2),
                'Preco_2026_min': round(preco_2026_min, 2),
                'Preco_2026_max': round(preco_2026_max, 2),
                'Crescimento_%': round(crescimento, 1)
            })
    except Exception as e:
        print(f"   ⚠️ Erro ao processar {uf}: {str(e)}")

# Salvar resultados
if resultados_2026:
    resultados_df = pd.DataFrame(resultados_2026)
    resultados_df = resultados_df.sort_values('Crescimento_%', ascending=False)
    resultados_df.to_csv('outputs/modelos/previsoes_2026.csv', index=False)
    
    print("\n📊 Resumo das Previsões (ordenado por crescimento):")
    print(resultados_df.to_string(index=False))
else:
    print("\n❌ Nenhum resultado disponível para salvar!")
    resultados_df = pd.DataFrame()  # DataFrame vazio para evitar erros

# ============================================
# 7. SALVAR MODELOS
# ============================================
print("\n💾 7. Salvando modelos...")

try:
    import joblib
    
    for uf, modelo in modelos.items():
        joblib.dump(modelo, f'outputs/modelos/prophet_modelo_{uf}.pkl')
    
    print("✅ Modelos Prophet salvos em: outputs/modelos/")
except Exception as e:
    print(f"⚠️ Erro ao salvar modelos: {str(e)}")
    print("   (Modelos não salvos, mas previsões geradas)")

# ============================================
# 8. RELATÓRIO FINAL
# ============================================
print("\n" + "="*80)
print("📊 RELATÓRIO FINAL - PREVISÕES PROPHET")
print("="*80)

if not resultados_df.empty:
    # Encontrar os destaques
    uf_maior_cresc = resultados_df.iloc[0]
    uf_menor_cresc = resultados_df.iloc[-1]
    
    # UF específica - CE (seu estado)
    ce_result = resultados_df[resultados_df['UF'] == 'CE']
    
    print(f"""
✅ PREVISÕES PARA 2026 CONCLUÍDAS!

📈 DESTAQUES:

1️⃣ Maior crescimento: {uf_maior_cresc['UF']}
   • Crescimento: {uf_maior_cresc['Crescimento_%']:.1f}%
   • Preço médio 2026: R${uf_maior_cresc['Preco_2026_medio']:,.2f}

2️⃣ Menor crescimento: {uf_menor_cresc['UF']}
   • Crescimento: {uf_menor_cresc['Crescimento_%']:.1f}%
   • Preço médio 2026: R${uf_menor_cresc['Preco_2026_medio']:,.2f}

3️⃣ Santa Catarina (SC)
   • Preço médio 2026: R${resultados_df[resultados_df['UF']=='SC']['Preco_2026_medio'].values[0]:,.2f}
   • Crescimento: {resultados_df[resultados_df['UF']=='SC']['Crescimento_%'].values[0]:.1f}%
""")
    
    if not ce_result.empty:
        print(f"""
4️⃣ Ceará (CE) - Seu estado! 🎉
   • Preço médio 2026: R${ce_result['Preco_2026_medio'].values[0]:,.2f}
   • Crescimento: {ce_result['Crescimento_%'].values[0]:.1f}%
   {'🚀' if ce_result['Crescimento_%'].values[0] > 5 else '📈'}
""")
    else:
        print("\n4️⃣ Ceará (CE) - Não foi possível gerar previsões para este estado")
    
    print(f"""
📁 ARQUIVOS GERADOS:
   • outputs/graficos/prophet_previsoes_2026.png
   • outputs/graficos/prophet_componentes_*.png
   • outputs/modelos/previsoes_2026.csv
   • outputs/modelos/prophet_modelo_*.pkl

🚀 PRÓXIMOS PASSOS:
   • Carregar modelos salvos para fazer novas previsões
   • Integrar com dados reais do FipeZAP
   • Criar dashboard com Streamlit
""")
else:
    print("""
❌ NENHUMA PREVISÃO FOI GERADA!

Possíveis causas:
   • Dados insuficientes para treinar os modelos
   • Erro durante o treinamento de todos os modelos
   • Problemas com a instalação do Prophet

Soluções:
   • Verifique se o arquivo dados_tratados.csv existe
   • Verifique se há dados suficientes (mínimo 12 meses por UF)
   • Instale o Prophet: pip install prophet
""")

print("\n" + "="*80)
print("✅ MODELO PROPHET CONCLUÍDO!")
print("="*80)