import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)

# 11 UFs + características especiais
ufs = {
    'CE': {'nome': 'Ceará', 'regiao': 'Nordeste', 'preco_base_2015': 340_000,
           'mcmv_peso': 0.14, 'crescimento_pos_crise': 1.15, 'violencia_base': 0.035,
           'evento_especial': 'turismo_sazonal'},
    'RJ': {'nome': 'Rio de Janeiro', 'regiao': 'Sudeste', 'preco_base_2015': 470_000,
           'mcmv_peso': 0.07, 'crescimento_pos_crise': 1.04, 'violencia_base': 0.14,
           'evento_especial': 'violencia_alta'},
    'SP': {'nome': 'São Paulo', 'regiao': 'Sudeste', 'preco_base_2015': 520_000,
           'mcmv_peso': 0.06, 'crescimento_pos_crise': 1.08, 'violencia_base': 0.07,
           'evento_especial': 'maior_mercado'},
    'BA': {'nome': 'Bahia', 'regiao': 'Nordeste', 'preco_base_2015': 290_000,
           'mcmv_peso': 0.10, 'crescimento_pos_crise': 1.06, 'violencia_base': 0.055,
           'evento_especial': 'informalidade_alta'},
    'MG': {'nome': 'Minas Gerais', 'regiao': 'Sudeste', 'preco_base_2015': 380_000,
           'mcmv_peso': 0.08, 'crescimento_pos_crise': 1.07, 'violencia_base': 0.045,
           'evento_especial': 'estavel'},
    'RS': {'nome': 'Rio Grande do Sul', 'regiao': 'Sul', 'preco_base_2015': 410_000,
           'mcmv_peso': 0.07, 'crescimento_pos_crise': 1.05, 'violencia_base': 0.05,
           'evento_especial': 'enchente_2024'},
    'SC': {'nome': 'Santa Catarina', 'regiao': 'Sul', 'preco_base_2015': 430_000,
           'mcmv_peso': 0.07, 'crescimento_pos_crise': 1.18, 'violencia_base': 0.04,
           'evento_especial': 'boom_migratorio'},
    'AM': {'nome': 'Amazonas (Manaus)', 'regiao': 'Norte', 'preco_base_2015': 360_000,
           'mcmv_peso': 0.09, 'crescimento_pos_crise': 1.05, 'violencia_base': 0.065,
           'evento_especial': 'isolamento_geografico'},
    'DF': {'nome': 'Distrito Federal', 'regiao': 'Centro-Oeste', 'preco_base_2015': 480_000,
           'mcmv_peso': 0.08, 'crescimento_pos_crise': 1.06, 'violencia_base': 0.05,
           'evento_especial': 'servidores_publicos'},
    'GO': {'nome': 'Goiás', 'regiao': 'Centro-Oeste', 'preco_base_2015': 330_000,
           'mcmv_peso': 0.09, 'crescimento_pos_crise': 1.10, 'violencia_base': 0.045,
           'evento_especial': 'expansao_agro'}
}

# Datas (2015-2025)
datas = pd.date_range('2015-01-01', '2025-12-01', freq='MS')

def get_governo(data):
    """Retorna o governo baseado na data (TIMESTAMP)"""
    # Converter para string no formato YYYY-MM-DD para comparação
    data_str = data.strftime('%Y-%m-%d')
    
    if data_str <= '2016-08-31':
        return 'Dilma'
    elif data_str <= '2018-12-31':
        return 'Temer'
    elif data_str <= '2022-12-31':
        return 'Bolsonaro'
    else:
        return 'Lula'

# Lista para acumular registros
records = []

print("🔄 Gerando dados...")
total_meses = len(datas)
total_ufs = len(ufs)

for idx, data in enumerate(datas):
    ano = data.year
    mes = data.month
    
    for uf, props in ufs.items():
        governo = get_governo(data)
        
        # --- PREÇO MÉDIO DOS IMÓVEIS ---
        preco_base = props['preco_base_2015']
        
        # Efeito crise 2015-2018 (Dilma/Temer)
        if governo in ['Dilma', 'Temer']:
            meses_crise = (ano - 2015) * 12 + mes
            fator_crise = 0.98 - 0.0025 * (meses_crise / 12)
        # Bolsonaro: pandemia e recuperação
        elif governo == 'Bolsonaro':
            if ano <= 2020:
                fator_crise = 0.94 + 0.01 * ((ano - 2019) * 12 + mes) / 12
            elif ano == 2021:
                fator_crise = 1.02 + 0.015 * (mes / 12)
            else:
                fator_crise = props['crescimento_pos_crise']
        # Lula: recuperação consolidada
        else:
            if uf in ['SC', 'CE', 'GO']:  # UFs que mais cresceram
                fator_crise = 1.12
            else:
                fator_crise = 1.05
        
        # Evento especial: Enchente no RS (maio 2024) - queda brusca nos preços
        if uf == 'RS' and ano == 2024 and mes >= 5:
            fator_crise = fator_crise * 0.85  # -15% por causa da enchente
        
        # Evento especial: Boom migratório SC (pós 2021)
        if uf == 'SC' and ano >= 2022:
            fator_crise = fator_crise * 1.08  # +8% por migração
        
        preco = preco_base * fator_crise * (1 + np.random.normal(0, 0.025))
        
        # --- INADIMPLÊNCIA NO ALUGUEL ---
        inad_base = 0.04
        if uf == 'RJ':
            inad_base = 0.065
        elif uf == 'CE':
            inad_base = 0.032
        
        if governo in ['Dilma', 'Temer']:
            inad = inad_base + 0.025 + (0.002 * ((ano - 2015) * 12 + mes) / 12)
        elif governo == 'Bolsonaro':
            inad = max(0.035, inad_base + 0.015 - (0.0015 * ((ano - 2019) * 12 + mes) / 12))
        else:
            inad = max(0.028, inad_base - 0.003)
        
        # Enchente RS aumenta inadimplência
        if uf == 'RS' and ano == 2024 and mes >= 5:
            inad = min(0.20, inad * 1.4)
        
        # Boom SC diminui inadimplência (mercado aquecido)
        if uf == 'SC' and ano >= 2022:
            inad = inad * 0.85
        
        inad = min(0.25, max(0.02, inad + np.random.normal(0, 0.004)))
        
        # --- TAXA DE OCUPAÇÃO (inverso de abandono) ---
        if governo in ['Dilma', 'Temer']:
            ocup = 0.91 - 0.002 * ((ano - 2015) * 12 + mes) / 12
        elif governo == 'Bolsonaro':
            ocup = 0.87 + 0.003 * ((ano - 2019) * 12 + mes) / 12
        else:
            ocup = 0.925
        
        # Enchente RS reduz ocupação
        if uf == 'RS' and ano == 2024 and mes >= 5:
            ocup = ocup * 0.88
        
        # Migração SC aumenta ocupação
        if uf == 'SC' and ano >= 2022:
            ocup = min(0.96, ocup * 1.04)
        
        ocup = min(0.97, max(0.75, ocup + np.random.normal(0, 0.007)))
        
        # --- VENDAS TOTAIS (por 10k hab) ---
        if governo in ['Dilma', 'Temer']:
            vendas = 23 - 0.08 * ((ano - 2015) * 12 + mes) / 12
        elif governo == 'Bolsonaro':
            if ano <= 2021:
                vendas = 19 + 0.5 * ((ano - 2019) * 12 + mes) / 12
            else:
                vendas = 25
        else:
            vendas = 27
        
        # Boom SC: vendas disparam
        if uf == 'SC' and ano >= 2022:
            vendas = vendas * 1.25
        
        # RS pós-enchente: vendas caem
        if uf == 'RS' and ano == 2024 and mes >= 5:
            vendas = vendas * 0.7
        
        vendas = max(8, vendas + np.random.normal(0, 1.2))
        
        # --- PERCENTUAL DE COMPRAS À VISTA ---
        if governo in ['Dilma', 'Temer']:
            perc_vista = 0.23 + np.random.normal(0, 0.02)
        else:
            perc_vista = 0.32 + np.random.normal(0, 0.02)
        
        perc_financ = 0.67 - perc_vista - (0.05 if governo in ['Dilma','Temer'] else 0.07)
        perc_financ = max(0.05, min(0.80, perc_financ))  # Manter entre 5% e 80%
        
        # --- MINHA CASA MINHA VIDA (MCMV) ---
        perc_mcmv = props['mcmv_peso']
        if governo == 'Lula':
            perc_mcmv = perc_mcmv * 1.35
        elif governo == 'Bolsonaro':
            perc_mcmv = perc_mcmv * 0.88
        
        # Enchente RS: governo libera MCMV para desabrigados
        if uf == 'RS' and ano == 2024 and mes >= 6:
            perc_mcmv = min(0.30, perc_mcmv * 1.5)
        
        perc_mcmv = min(0.28, max(0.04, perc_mcmv + np.random.normal(0, 0.008)))
        
        # --- COMPRA DE IMÓVEL USADO (com antigo dono) ---
        perc_usado = 0.70 + (0.06 if governo in ['Dilma','Temer'] else 0)
        if uf == 'SC' and ano >= 2022:  # Mercado novo aquecido
            perc_usado = perc_usado * 0.92
        perc_usado = min(0.85, max(0.50, perc_usado + np.random.normal(0, 0.01)))
        
        # --- COMPRA DE TERRENO DO ZERO ---
        perc_terreno_zero = 0.08 + (0.03 if uf in ['CE', 'SC', 'GO'] else 0)
        if governo == 'Lula':
            perc_terreno_zero = perc_terreno_zero * 1.1
        perc_terreno_zero = min(0.18, max(0.03, perc_terreno_zero + np.random.normal(0, 0.005)))
        
        # --- ABANDONO POR VIOLÊNCIA ---
        aband_viol_base = props['violencia_base']
        if governo in ['Dilma','Temer']:
            aband_viol = aband_viol_base * (1 + 0.06 * ((ano - 2015) * 12 + mes) / 12)
        elif governo == 'Bolsonaro':
            aband_viol = aband_viol_base * (0.92 if ano >= 2021 else 1.04)
        else:
            aband_viol = aband_viol_base * 0.87
        
        # RJ tem pico de violência em 2017-2018
        if uf == 'RJ' and ano in [2017, 2018]:
            aband_viol = aband_viol * 1.25
        
        aband_viol = min(0.35, max(0.01, aband_viol + np.random.normal(0, 0.008)))
        
        # --- ABANDONO POR DESASTRE NATURAL (enchente RS 2024) ---
        aband_desastre = 0
        if uf == 'RS' and ano == 2024 and mes >= 5:
            aband_desastre = 0.08 * (1 + 0.1 * (mes - 4))  # Cresce após a enchente
            aband_desastre = min(0.15, aband_desastre)
        
        # --- ABANDONO POR VONTADE PRÓPRIA (mudança) ---
        aband_vontade = 0.03 * (1 + 0.01 * ((ano - 2015) * 12 + mes) / 12)
        if uf == 'SC' and ano >= 2022:  # Menos abandono, mais gente chegando
            aband_vontade = aband_vontade * 0.7
        aband_vontade = max(0.01, min(0.08, aband_vontade))
        
        # --- AUMENTO DE MORADORES (migração) - NOVO INDICADOR! ---
        aumento_moradores = 0
        if uf == 'SC' and ano >= 2022:
            aumento_moradores = 0.025 * (1 + 0.08 * ((ano - 2022) * 12 + mes) / 12)
            aumento_moradores = min(0.06, aumento_moradores)
        elif uf == 'GO' and ano >= 2023:
            aumento_moradores = 0.012 * (1 + 0.05 * ((ano - 2023) * 12 + mes) / 12)
        elif uf == 'CE' and ano >= 2023:
            aumento_moradores = 0.008 * (1 + 0.04 * ((ano - 2023) * 12 + mes) / 12)
        
        # --- TERRENOS INVADIDOS / FAVELIZAÇÃO - NOVO INDICADOR! ---
        invasao_base = {
            'SP': 0.012, 'RJ': 0.025, 'CE': 0.008, 'BA': 0.018,
            'MG': 0.007, 'RS': 0.006, 'SC': 0.003, 'AM': 0.015,
            'DF': 0.009, 'GO': 0.007
        }.get(uf, 0.008)
        
        invasao_terrenos = invasao_base
        # Crise aumenta invasões
        if governo in ['Dilma', 'Temer']:
            invasao_terrenos = invasao_terrenos * (1 + 0.08 * ((ano - 2015) * 12 + mes) / 12)
        # RJ tem histórico de favelização mais intensa
        if uf == 'RJ':
            invasao_terrenos = invasao_terrenos * 1.3
        # Enchente RS causa invasões em áreas altas
        if uf == 'RS' and ano == 2024 and mes >= 6:
            invasao_terrenos = min(0.04, invasao_terrenos * 1.4)
        
        invasao_terrenos = min(0.05, max(0.001, invasao_terrenos + np.random.normal(0, 0.001)))
        
        # --- QUALIDADE DE VIDA ---
        qv_base = {
            'SC': 7.2, 'SP': 7.0, 'DF': 7.1, 'RS': 6.9,
            'MG': 6.7, 'CE': 6.4, 'GO': 6.5, 'BA': 6.0,
            'AM': 6.2, 'RJ': 5.9
        }[uf]
        
        if governo == 'Dilma':
            qv = qv_base - 0.02 * ((ano - 2015) * 12 + mes) / 12
        elif governo == 'Temer':
            qv = qv_base - 0.03 * ((ano - 2016) * 12 + mes) / 12
        elif governo == 'Bolsonaro':
            qv = qv_base + 0.04 * ((ano - 2019) * 12 + mes) / 12
        else:
            qv = qv_base + 0.09 * ((ano - 2023) * 12 + mes) / 12
        
        # Enchente RS degrada qualidade de vida temporariamente
        if uf == 'RS' and ano == 2024 and mes >= 5:
            qv = qv - 0.6
        
        # SC melhorando rápido
        if uf == 'SC' and ano >= 2022:
            qv = qv + 0.08
        
        qv = max(4.5, min(8.5, qv + np.random.normal(0, 0.08)))
        
        # --- PESSOAS QUE COLOCAM PRÓPRIO IMÓVEL PARA ALUGAR ---
        aluga_proprio = 0.7 + (0.5 if governo in ['Dilma','Temer'] else 0)
        if uf == 'CE' and mes in [12, 1, 2, 7]:  # Temporada
            aluga_proprio = aluga_proprio * 1.3
        aluga_proprio = max(0.2, aluga_proprio + np.random.normal(0, 0.1))
        
        # --- PESSOAS QUE VENDEM PRÓPRIO IMÓVEL ---
        vende_proprio = 0.55 + (0.4 if governo in ['Dilma','Temer'] else 0)
        vende_proprio = max(0.2, vende_proprio + np.random.normal(0, 0.08))
        
        # --- ÍNDICE DE ALUGUEL (variação anual) ---
        if governo == 'Bolsonaro' and ano == 2021:
            aluguel_yoy = 0.085 + np.random.normal(0, 0.008)
        else:
            aluguel_yoy = 0.05 + np.random.normal(0, 0.007)
        
        # SC aluguel dispara por demanda
        if uf == 'SC' and ano >= 2022:
            aluguel_yoy = aluguel_yoy + 0.025
        
        aluguel_yoy = max(0.01, min(0.15, aluguel_yoy))
        
        # Converter para inteiros/per capita (multiplicar por 1000 para evitar floats muito pequenos)
        records.append([
            data.strftime('%Y-%m'), uf, props['regiao'], governo,
            round(preco, -2), round(ocup * 100, 1),
            round(inad * 100, 1), round(vendas, 1),
            round(perc_vista * 100, 1), round(perc_financ * 100, 1),
            round(perc_mcmv * 100, 1), round(perc_usado * 100, 1),
            round(perc_terreno_zero * 100, 1), round(qv, 2),
            round(aband_viol * 1000, 1), round(aband_desastre * 1000, 1),
            round(aband_vontade * 1000, 1), round(aumento_moradores * 1000, 1),
            round(invasao_terrenos * 1000, 1), round(aluga_proprio, 1),
            round(vende_proprio, 1), round(aluguel_yoy * 100, 1)
        ])
    
    # Mostrar progresso
    if (idx + 1) % 24 == 0:  # A cada 2 anos
        print(f"✅ Processados {idx + 1} de {total_meses} meses...")

# Criar DataFrame
print("\n📊 Criando DataFrame...")
df = pd.DataFrame(records, columns=[
    'data', 'uf', 'regiao', 'governo', 'preco_medio_imovel',
    'tx_ocupacao_%', 'inadimplencia_aluguel_%', 'vendas_totais_10k_hab',
    'perc_vista_%', 'perc_financiamento_%', 'perc_mcmv_%', 'perc_imovel_usado_%',
    'perc_terreno_zero_%', 'qualidade_vida_score',
    'abandono_violencia_por_10k', 'abandono_desastre_por_10k',
    'abandono_vontade_por_10k', 'aumento_moradores_por_10k',
    'invasao_terrenos_por_10k', 'pessoas_alugam_proprio_10k',
    'pessoas_vendem_proprio_10k', 'indice_aluguel_yoy_%'
])

# Salvar
print("💾 Salvando arquivo CSV...")
df.to_csv('dados_imobiliarios_brasil_10estados_2015_2025.csv', index=False)

print(f"\n✅ DATASET GERADO COM SUCESSO!")
print(f"📊 Total de registros: {len(df):,}")
print(f"📍 Total de UFs: {df['uf'].nunique()}")
print(f"📅 Período: {df['data'].min()} a {df['data'].max()}")

# Estatísticas por UF
print("\n" + "="*80)
print("📈 RESUMO POR UF (2015-2025)")
print("="*80)

resumo = df.groupby('uf').agg({
    'preco_medio_imovel': 'mean',
    'perc_mcmv_%': 'mean',
    'inadimplencia_aluguel_%': 'mean',
    'abandono_violencia_por_10k': 'mean',
    'invasao_terrenos_por_10k': 'mean',
    'aumento_moradores_por_10k': 'mean',
    'qualidade_vida_score': 'mean'
}).round(2)

# Ordenar por preço médio
resumo = resumo.sort_values('preco_medio_imovel', ascending=False)

print(resumo)

# Casos especiais
print("\n" + "="*80)
print("🎯 EVENTOS ESPECIAIS")
print("="*80)

rs_enchente = df[(df['uf']=='RS') & (df['data']>='2024-05')]
print(f"🌊 RS pós-enchente (2024): {len(rs_enchente)} meses de impacto")
print(f"   - Queda de preço: {((rs_enchente['preco_medio_imovel'].iloc[0] - rs_enchente['preco_medio_imovel'].iloc[-1]) / rs_enchente['preco_medio_imovel'].iloc[0] * 100):.1f}%")

sc_boom = df[df['uf']=='SC']
print(f"🚀 SC boom migratório: aumento médio de moradores = {sc_boom['aumento_moradores_por_10k'].mean():.2f} por 10k hab")
print(f"   - Crescimento de preços pós-2021: +{((sc_boom[sc_boom['data']>='2022-01']['preco_medio_imovel'].mean() / sc_boom[sc_boom['data']<'2022-01']['preco_medio_imovel'].mean() - 1) * 100):.1f}%")

print(f"🏘️ RJ invasões: média = {df[df['uf']=='RJ']['invasao_terrenos_por_10k'].mean():.2f} por 10k hab")
print(f"💚 CE (seu estado!): crescimento MCMV = {df[df['uf']=='CE']['perc_mcmv_%'].mean():.1f}%")

print("\n✅ Arquivo salvo: dados_imobiliarios_brasil_10estados_2015_2025.csv")