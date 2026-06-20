"""
FASE 1: TRATAMENTO DE DADOS
Objetivo: Limpar, transformar e preparar os dados para análise
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🏠 PROJETO IMOBILIÁRIO BRASIL - FASE 1: TRATAMENTO DE DADOS")
print("="*80)

# 1. CARREGAR DADOS
print("\n📂 1. Carregando dados...")
df = pd.read_csv('data/dados_imobiliarios_brasil_10estados_2015_2025.csv')

print(f"✅ Dados carregados: {len(df)} registros, {len(df.columns)} colunas")

# 2. ANÁLISE INICIAL
print("\n🔍 2. Análise inicial dos dados...")
print(f"\n📊 Primeiras 5 linhas:")
print(df.head())

print(f"\n📋 Informações do dataset:")
print(df.info())

print(f"\n📈 Estatísticas descritivas básicas:")
print(df.describe())

# 3. VERIFICAR DADOS FALTANTES
print("\n🔎 3. Verificando dados faltantes...")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Coluna': missing.index,
    'Valores Faltantes': missing.values,
    'Percentual': missing_percent.values
})
print(missing_df[missing_df['Valores Faltantes'] > 0])

if missing.sum() == 0:
    print("✅ Nenhum dado faltante encontrado!")
else:
    print("⚠️ Dados faltantes encontrados - tratando...")

# 4. CONVERSÃO DE TIPOS
print("\n🔄 4. Convertendo tipos de dados...")

# Converter data para datetime
df['data'] = pd.to_datetime(df['data'])

# Extrair componentes de data
df['ano'] = df['data'].dt.year
df['mes'] = df['data'].dt.month
df['trimestre'] = df['data'].dt.quarter
df['semestre'] = df['data'].dt.month.apply(lambda x: 1 if x <= 6 else 2)

print("✅ Tipos convertidos:")
print(df[['data', 'ano', 'mes', 'trimestre', 'semestre']].head())

# 5. CRIAR VARIÁVEIS DERIVADAS
print("\n🔧 5. Criando variáveis derivadas...")

# Índice de Crise (0-1): baseado na inadimplência e ocupação
df['indice_crise'] = (df['inadimplencia_aluguel_%'] / 25) + (1 - df['tx_ocupacao_%'] / 100)
df['indice_crise'] = df['indice_crise'] / 2  # Normalizar

# Variação de preço (ano anterior)
df = df.sort_values(['uf', 'data'])
df['preco_ano_anterior'] = df.groupby('uf')['preco_medio_imovel'].shift(12)
df['variacao_preco_anual'] = (df['preco_medio_imovel'] / df['preco_ano_anterior'] - 1) * 100

# Abandono total
df['abandono_total_por_10k'] = df['abandono_violencia_por_10k'] + \
                               df['abandono_desastre_por_10k'] + \
                               df['abandono_vontade_por_10k']

# Rotacionamento de imóveis (vendas + aluguel)
df['rotacao_imoveis'] = df['vendas_totais_10k_hab'] + df['pessoas_alugam_proprio_10k']

# Score de saúde do mercado (0-100)
df['saude_mercado'] = 100 - (df['inadimplencia_aluguel_%'] * 2) - \
                      (df['abandono_total_por_10k'] * 3) + \
                      (df['tx_ocupacao_%'] * 0.5) + \
                      (df['qualidade_vida_score'] * 5)

# Normalizar score
df['saude_mercado'] = df['saude_mercado'].clip(0, 100)

print("✅ Variáveis derivadas criadas:")
derivadas = ['indice_crise', 'variacao_preco_anual', 'abandono_total_por_10k', 
             'rotacao_imoveis', 'saude_mercado']
print(df[derivadas].head())

# 6. FILTRAR OUTLIERS
print("\n🚫 6. Identificando e tratando outliers...")

def detect_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)

# Identificar outliers nas principais variáveis numéricas
colunas_numericas = ['preco_medio_imovel', 'inadimplencia_aluguel_%', 
                     'vendas_totais_10k_hab', 'qualidade_vida_score']

outliers_count = {}
for col in colunas_numericas:
    outliers = detect_outliers_iqr(df[col])
    outliers_count[col] = outliers.sum()
    print(f"   {col}: {outliers.sum()} outliers identificados")

# Decisão: vamos manter os outliers (são dados reais)
print("✅ Outliers mantidos (representam eventos reais como enchentes e crises)")

# 7. SALVAR DADOS TRATADOS
print("\n💾 7. Salvando dados tratados...")

# Selecionar colunas organizadas
colunas_ordenadas = [
    'data', 'ano', 'mes', 'trimestre', 'semestre',
    'uf', 'regiao', 'governo',
    'preco_medio_imovel', 'variacao_preco_anual',
    'tx_ocupacao_%', 'inadimplencia_aluguel_%',
    'vendas_totais_10k_hab', 'rotacao_imoveis',
    'perc_vista_%', 'perc_financiamento_%', 'perc_mcmv_%',
    'perc_imovel_usado_%', 'perc_terreno_zero_%',
    'indice_aluguel_yoy_%',
    'qualidade_vida_score', 'indice_crise', 'saude_mercado',
    'abandono_violencia_por_10k', 'abandono_desastre_por_10k',
    'abandono_vontade_por_10k', 'abandono_total_por_10k',
    'aumento_moradores_por_10k', 'invasao_terrenos_por_10k',
    'pessoas_alugam_proprio_10k', 'pessoas_vendem_proprio_10k'
]

df_tratado = df[colunas_ordenadas]

# Salvar
df_tratado.to_csv('outputs/dados_tratados.csv', index=False)

print(f"✅ Dados tratados salvos: {len(df_tratado)} registros, {len(df_tratado.columns)} colunas")
print(f"📁 Arquivo: outputs/dados_tratados.csv")

# 8. RELATÓRIO FINAL
print("\n" + "="*80)
print("📊 RELATÓRIO DE TRATAMENTO")
print("="*80)

print(f"""
✅ TRATAMENTO CONCLUÍDO COM SUCESSO!

📋 Resumo do Dataset:
   - Total de registros: {len(df_tratado):,}
   - Total de colunas: {len(df_tratado.columns)}
   - Período: {df_tratado['data'].min()} a {df_tratado['data'].max()}
   - UFs: {', '.join(df_tratado['uf'].unique())}
   - Governos: {', '.join(df_tratado['governo'].unique())}

📊 Variáveis Criadas:
   ✓ indice_crise (0-1): Mede o nível de crise do mercado
   ✓ variacao_preco_anual (%): Variação anual de preços
   ✓ abandono_total_por_10k: Soma de todos os tipos de abandono
   ✓ rotacao_imoveis: Indicador de movimento do mercado
   ✓ saude_mercado (0-100): Score geral de saúde do mercado

🎯 Próxima Fase: Análise Estatística
   - Executar: analise_estatistica.py
   - Análises: correlações, testes de hipótese, tendências
""")

# Mostrar amostra final
print("\n📋 Amostra dos dados tratados:")
print(df_tratado[['data', 'uf', 'governo', 'preco_medio_imovel', 
                  'inadimplencia_aluguel_%', 'saude_mercado']].head(10))

print("\n✅ FASE 1 CONCLUÍDA!")