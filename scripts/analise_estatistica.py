"""
FASE 2: ANÁLISE ESTATÍSTICA
Objetivo: Realizar análises estatísticas profundas e testes de hipótese
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway, ttest_ind
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("📊 PROJETO IMOBILIÁRIO BRASIL - FASE 2: ANÁLISE ESTATÍSTICA")
print("="*80)

# 1. CARREGAR DADOS TRATADOS
print("\n📂 1. Carregando dados tratados...")
df = pd.read_csv('outputs/dados_tratados.csv')
df['data'] = pd.to_datetime(df['data'])

print(f"✅ Dados carregados: {len(df)} registros")

# 2. ESTATÍSTICAS DESCRITIVAS POR CATEGORIA
print("\n📈 2. Estatísticas descritivas por governo...")

estatisticas_governo = df.groupby('governo').agg({
    'preco_medio_imovel': ['mean', 'median', 'std', 'min', 'max'],
    'inadimplencia_aluguel_%': ['mean', 'median', 'std'],
    'qualidade_vida_score': ['mean', 'std'],
    'saude_mercado': ['mean', 'std'],
    'vendas_totais_10k_hab': ['mean', 'std']
}).round(2)

print(estatisticas_governo)

# Salvar
estatisticas_governo.to_csv('outputs/estatisticas_por_governo.csv')
print("✅ Estatísticas por governo salvas")

# 3. ESTATÍSTICAS POR UF
print("\n📊 3. Estatísticas descritivas por UF...")

estatisticas_uf = df.groupby('uf').agg({
    'preco_medio_imovel': ['mean', 'std'],
    'inadimplencia_aluguel_%': ['mean', 'std'],
    'perc_mcmv_%': ['mean', 'std'],
    'qualidade_vida_score': ['mean', 'std'],
    'saude_mercado': ['mean', 'std'],
    'invasao_terrenos_por_10k': ['mean', 'std']
}).round(2)

# Ordenar por preço médio
estatisticas_uf = estatisticas_uf.sort_values(('preco_medio_imovel', 'mean'), ascending=False)
print(estatisticas_uf)

estatisticas_uf.to_csv('outputs/estatisticas_por_uf.csv')
print("✅ Estatísticas por UF salvas")

# 4. ANÁLISE DE CORRELAÇÃO
print("\n🔗 4. Análise de correlação entre variáveis...")

# Selecionar variáveis numéricas para correlação
variaveis_correlacao = [
    'preco_medio_imovel', 'tx_ocupacao_%', 'inadimplencia_aluguel_%',
    'vendas_totais_10k_hab', 'perc_mcmv_%', 'qualidade_vida_score',
    'indice_crise', 'saude_mercado', 'aumento_moradores_por_10k',
    'invasao_terrenos_por_10k', 'abandono_total_por_10k'
]

# Matriz de correlação
corr_matrix = df[variaveis_correlacao].corr()

print("\n📊 Matriz de correlação (principais correlações):")
# Mostrar correlações fortes (>0.5 ou <-0.5)
corr_strong = corr_matrix[(corr_matrix > 0.5) | (corr_matrix < -0.5)]
print(corr_strong)

# Salvar matriz completa
corr_matrix.to_csv('outputs/matriz_correlacao.csv')
print("✅ Matriz de correlação salva")

# 5. TESTES DE HIPÓTESE
print("\n🧪 5. Testes de hipótese...")

# Teste 1: Preço médio difere entre governos?
print("\n🎯 Teste 1: Preço médio difere entre governos?")
governos = df['governo'].unique()
precos_por_governo = [df[df['governo'] == gov]['preco_medio_imovel'] for gov in governos]
f_stat, p_value = f_oneway(*precos_por_governo)

print(f"   ANOVA: F-statistic = {f_stat:.3f}, p-value = {p_value:.6f}")
if p_value < 0.05:
    print("   ✅ Há diferença significativa entre os governos (p < 0.05)")
else:
    print("   ❌ Não há diferença significativa entre os governos")

# Teste 2: Inadimplência difere entre Dilma/Temer vs Lula?
print("\n🎯 Teste 2: Inadimplência difere entre crise (Dilma/Temer) e recuperação (Lula)?")
crise = df[df['governo'].isin(['Dilma', 'Temer'])]['inadimplencia_aluguel_%']
recuperacao = df[df['governo'] == 'Lula']['inadimplencia_aluguel_%']
t_stat, p_value = ttest_ind(crise, recuperacao)

print(f"   T-test: t-statistic = {t_stat:.3f}, p-value = {p_value:.6f}")
if p_value < 0.05:
    print("   ✅ Há diferença significativa entre crise e recuperação")
else:
    print("   ❌ Não há diferença significativa")

# Teste 3: Correlação entre qualidade de vida e preço
print("\n🎯 Teste 3: Correlação entre qualidade de vida e preço dos imóveis?")
corr, p_value = pearsonr(df['qualidade_vida_score'], df['preco_medio_imovel'])
print(f"   Pearson correlation: {corr:.3f}, p-value: {p_value:.6f}")
if p_value < 0.05:
    print(f"   ✅ Correlação significativa: {corr:.2f}")
else:
    print("   ❌ Correlação não significativa")

# 6. ANÁLISE DE SÉRIE TEMPORAL
print("\n⏰ 6. Análise de série temporal...")

# Tendência de preços por UF
tendencia = df.groupby(['uf', 'ano'])['preco_medio_imovel'].mean().unstack()

# Calcular crescimento anual médio
crescimento_anual = {}
for uf in tendencia.index:
    if len(tendencia.loc[uf]) > 1:
        crescimento = ((tendencia.loc[uf].iloc[-1] / tendencia.loc[uf].iloc[0]) ** (1/10) - 1) * 100
        crescimento_anual[uf] = crescimento

crescimento_df = pd.DataFrame(list(crescimento_anual.items()), 
                              columns=['UF', 'Crescimento_Anual_%'])
crescimento_df = crescimento_df.sort_values('Crescimento_Anual_%', ascending=False)

print("\n📈 Crescimento anual médio por UF (2015-2025):")
print(crescimento_df)

crescimento_df.to_csv('outputs/crescimento_anual_por_uf.csv', index=False)
print("✅ Crescimento anual por UF salvo")

# 7. ANÁLISE DE EVENTOS ESPECIAIS
print("\n🎯 7. Análise de eventos especiais...")

# Enchente RS (2024)
rs_antes = df[(df['uf']=='RS') & (df['ano'] < 2024)]
rs_depois = df[(df['uf']=='RS') & (df['ano'] == 2024) & (df['mes'] >= 5)]

if len(rs_depois) > 0:
    print("\n🌊 Impacto da Enchente no RS (2024):")
    print(f"   Preço médio antes: R${rs_antes['preco_medio_imovel'].mean():.2f}")
    print(f"   Preço médio depois: R${rs_depois['preco_medio_imovel'].mean():.2f}")
    print(f"   Variação: {((rs_depois['preco_medio_imovel'].mean() / rs_antes['preco_medio_imovel'].mean()) - 1) * 100:.1f}%")
    print(f"   Inadimplência antes: {rs_antes['inadimplencia_aluguel_%'].mean():.1f}%")
    print(f"   Inadimplência depois: {rs_depois['inadimplencia_aluguel_%'].mean():.1f}%")

# Boom SC (2022+)
sc_antes = df[(df['uf']=='SC') & (df['ano'] < 2022)]
sc_depois = df[(df['uf']=='SC') & (df['ano'] >= 2022)]

print("\n🚀 Boom Migratório em SC (2022+):")
print(f"   Preço médio antes: R${sc_antes['preco_medio_imovel'].mean():.2f}")
print(f"   Preço médio depois: R${sc_depois['preco_medio_imovel'].mean():.2f}")
print(f"   Variação: {((sc_depois['preco_medio_imovel'].mean() / sc_antes['preco_medio_imovel'].mean()) - 1) * 100:.1f}%")
print(f"   Aumento de moradores: {sc_depois['aumento_moradores_por_10k'].mean():.2f} por 10k hab")
print(f"   Variação de vendas: {((sc_depois['vendas_totais_10k_hab'].mean() / sc_antes['vendas_totais_10k_hab'].mean()) - 1) * 100:.1f}%")

# 8. RELATÓRIO FINAL
print("\n" + "="*80)
print("📊 RELATÓRIO DE ANÁLISE ESTATÍSTICA")
print("="*80)

print(f"""
✅ ANÁLISE ESTATÍSTICA CONCLUÍDA!

📊 Principais Insights Estatísticos:

1. Governo com maior preço médio: 
   {estatisticas_governo['preco_medio_imovel']['mean'].idxmax()} 
   (R${estatisticas_governo['preco_medio_imovel']['mean'].max():,.2f})

2. UF com maior preço médio: 
   {estatisticas_uf['preco_medio_imovel']['mean'].idxmax()} 
   (R${estatisticas_uf['preco_medio_imovel']['mean'].max():,.2f})

3. UF com maior crescimento anual: 
   {crescimento_df.iloc[0]['UF']} ({crescimento_df.iloc[0]['Crescimento_Anual_%']:.1f}% ao ano)

4. Correlação qualidade_vida × preço: {corr:.2f}

5. Diferença significativa entre governos: {'Sim' if p_value < 0.05 else 'Não'}

📁 Arquivos gerados:
   - estatisticas_por_governo.csv
   - estatisticas_por_uf.csv
   - matriz_correlacao.csv
   - crescimento_anual_por_uf.csv

🎯 Próxima Fase: Análise Gráfica
   - Executar: visualizacoes.py
   - Gráficos: evolução, comparações, heatmaps
""")

print("\n✅ FASE 2 CONCLUÍDA!")