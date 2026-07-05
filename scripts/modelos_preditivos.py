import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    classification_report, confusion_matrix, accuracy_score
)

# Criar diretórios
import os
os.makedirs('../outputs/graficos', exist_ok=True)
os.makedirs('../outputs/modelos', exist_ok=True)

print("="*80)
print("🤖 PROJETO IMOBILIÁRIO BRASIL - MODELOS PREDITIVOS")
print("="*80)

# ============================================
# 1. CARREGAR DADOS
# ============================================
print("\n📂 1. Carregando dados...")
df = pd.read_csv('../outputs/dados_tratados.csv')
df['data'] = pd.to_datetime(df['data'])

print(f"✅ Dados carregados: {len(df)} registros")
print(f"📅 Período: {df['data'].min()} a {df['data'].max()}")
print(f"📍 UFs: {df['uf'].nunique()}")

# ============================================
# 2. PREPARAR DADOS PARA MODELOS
# ============================================
print("\n🔧 2. Preparando dados...")

df_ml = df.copy()

# Codificar variáveis categóricas
le_uf = LabelEncoder()
le_governo = LabelEncoder()
le_regiao = LabelEncoder()

df_ml['uf_encoded'] = le_uf.fit_transform(df_ml['uf'])
df_ml['governo_encoded'] = le_governo.fit_transform(df_ml['governo'])
df_ml['regiao_encoded'] = le_regiao.fit_transform(df_ml['regiao'])

# Mapeamentos para referência
uf_mapping = dict(zip(le_uf.classes_, le_uf.transform(le_uf.classes_)))
governo_mapping = dict(zip(le_governo.classes_, le_governo.transform(le_governo.classes_)))

print(f"   UFs codificadas: {uf_mapping}")
print(f"   Governos codificados: {governo_mapping}")

# Features para modelos
feature_cols = [
    'ano', 'mes', 'trimestre', 'semestre',
    'uf_encoded', 'regiao_encoded', 'governo_encoded',
    'tx_ocupacao_%', 'vendas_totais_10k_hab',
    'perc_vista_%', 'perc_financiamento_%', 'perc_mcmv_%',
    'perc_imovel_usado_%', 'perc_terreno_zero_%',
    'indice_aluguel_yoy_%', 'qualidade_vida_score',
    'aumento_moradores_por_10k', 'invasao_terrenos_por_10k',
    'abandono_total_por_10k', 'rotacao_imoveis'
]

print(f"✅ Features preparadas: {len(feature_cols)}")

# ============================================
# 3. MODELO 1: PREVISÃO DE PREÇOS
# ============================================
print("\n" + "="*80)
print("📊 MODELO 1: PREVISÃO DE PREÇOS DOS IMÓVEIS")
print("="*80)

# Preparar dados
X = df_ml[feature_cols]
y = df_ml['preco_medio_imovel']

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalonar dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"📊 Dados preparados:")
print(f"   Treino: {X_train.shape[0]} registros")
print(f"   Teste: {X_test.shape[0]} registros")

# 3.1 Regressão Linear
print("\n🔵 Treinando Regressão Linear...")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"   MAE: R${mae_lr:,.2f}")
print(f"   RMSE: R${rmse_lr:,.2f}")
print(f"   R²: {r2_lr:.4f}")

# 3.2 Random Forest Regressor
print("\n🟢 Treinando Random Forest Regressor...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"   MAE: R${mae_rf:,.2f}")
print(f"   RMSE: R${rmse_rf:,.2f}")
print(f"   R²: {r2_rf:.4f}")

# Feature Importance
importance_rf = pd.DataFrame({
    'feature': feature_cols,
    'importancia': rf.feature_importances_
}).sort_values('importancia', ascending=False)

print("\n📈 Top 10 features mais importantes (Preço):")
print(importance_rf.head(10))

# Visualização 1: Preços Reais vs Preditos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico 1: Random Forest
axes[0].scatter(y_test, y_pred_rf, alpha=0.5, color='green')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2)
axes[0].set_xlabel('Preço Real (R$)', fontsize=12)
axes[0].set_ylabel('Preço Predito (R$)', fontsize=12)
axes[0].set_title(f'Random Forest - R² = {r2_rf:.4f}', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Comparação de Erros
erros = pd.DataFrame({
    'Linear': y_test - y_pred_lr,
    'Random Forest': y_test - y_pred_rf
})
erros.boxplot(ax=axes[1])
axes[1].set_title('Distribuição dos Erros', fontweight='bold')
axes[1].set_ylabel('Erro (R$)', fontsize=12)
axes[1].grid(True, alpha=0.3)

# Gráfico 3: Feature Importance
top_features = importance_rf.head(10)
axes[2].barh(top_features['feature'], top_features['importancia'], color='green')
axes[2].set_xlabel('Importância', fontsize=12)
axes[2].set_title('Top 10 Features - Preços', fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Modelo 1: Previsão de Preços dos Imóveis', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/graficos/modelo_precos_resultados.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# 4. MODELO 2: PREVISÃO DE INADIMPLÊNCIA
# ============================================
print("\n" + "="*80)
print("📊 MODELO 2: PREVISÃO DE INADIMPLÊNCIA")
print("="*80)

# Preparar dados
X2 = df_ml[feature_cols]
y2 = df_ml['inadimplencia_aluguel_%']

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

print(f"📊 Dados preparados:")
print(f"   Treino: {X2_train.shape[0]} registros")
print(f"   Teste: {X2_test.shape[0]} registros")

# Random Forest para Inadimplência
print("\n🟢 Treinando Random Forest para Inadimplência...")
rf2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf2.fit(X2_train, y2_train)
y2_pred = rf2.predict(X2_test)

mae2 = mean_absolute_error(y2_test, y2_pred)
rmse2 = np.sqrt(mean_squared_error(y2_test, y2_pred))
r2_2 = r2_score(y2_test, y2_pred)

print(f"   MAE: {mae2:.4f}%")
print(f"   RMSE: {rmse2:.4f}%")
print(f"   R²: {r2_2:.4f}")

# Feature Importance
importance2 = pd.DataFrame({
    'feature': feature_cols,
    'importancia': rf2.feature_importances_
}).sort_values('importancia', ascending=False)

print("\n📈 Top 10 features para prever inadimplência:")
print(importance2.head(10))

# Visualização
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gráfico 1: Real vs Predito
axes[0].scatter(y2_test, y2_pred, alpha=0.5, color='orange')
axes[0].plot([y2_test.min(), y2_test.max()], [y2_test.min(), y2_test.max()], 
             'r--', linewidth=2)
axes[0].set_xlabel('Inadimplência Real (%)', fontsize=12)
axes[0].set_ylabel('Inadimplência Predita (%)', fontsize=12)
axes[0].set_title(f'Random Forest - R² = {r2_2:.4f}', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Gráfico 2: Distribuição dos Erros
erros2 = y2_test - y2_pred
axes[1].hist(erros2, bins=30, color='orange', alpha=0.7, edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Erro (%)', fontsize=12)
axes[1].set_ylabel('Frequência', fontsize=12)
axes[1].set_title('Distribuição dos Erros', fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Gráfico 3: Feature Importance
top_features2 = importance2.head(10)
axes[2].barh(top_features2['feature'], top_features2['importancia'], color='orange')
axes[2].set_xlabel('Importância', fontsize=12)
axes[2].set_title('Top 10 Features - Inadimplência', fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Modelo 2: Previsão de Inadimplência', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../outputs/graficos/modelo_inadimplencia_resultados.png', dpi=300, bbox_inches='tight')
plt.show()
