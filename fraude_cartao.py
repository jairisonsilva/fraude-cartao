# DETECÇÃO DE FRAUDES EM TRANSAÇÕES DE CARTÃO

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from xgboost import XGBClassifier

import shap


warnings.filterwarnings("ignore")

# 2. CONFIGURAÇÕES

URL_DATASET = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "creditcard.csv"
)

RANDOM_STATE = 42

# 3. CARREGAMENTO DOS DADOS

print("\n" + "=" * 60)
print("CARREGANDO DATASET")
print("=" * 60)

df = pd.read_csv(URL_DATASET)

print(f"\nDataset carregado com sucesso!")
print(f"Quantidade de registros: {len(df):,}")
print(f"Quantidade de colunas: {len(df.columns)}")

# 4. EXPLORAÇÃO DOS DADOS

print("\n" + "=" * 60)
print("EXPLORAÇÃO DOS DADOS")
print("=" * 60)

print("\nPrimeiras linhas:")
print(df.head())

print("\nInformações:")
print(df.info())

print("\nEstatísticas:")
print(df.describe())

print("\nValores nulos:")
print(df.isnull().sum().sum())


# 5. ANÁLISE DO DESBALANCEAMENTO

print("\n" + "=" * 60)
print("ANÁLISE DO DESBALANCEAMENTO")
print("=" * 60)

quantidade_classes = df["Class"].value_counts()

normais = quantidade_classes[0]
fraudes = quantidade_classes[1]
total = len(df)

percentual_normais = normais / total * 100
percentual_fraudes = fraudes / total * 100

print(f"\nTransações normais: {normais:,}")
print(f"Percentual de normais: {percentual_normais:.2f}%")

print(f"\nTransações fraudulentas: {fraudes:,}")
print(f"Percentual de fraudes: {percentual_fraudes:.2f}%")


# Gráfico da distribuição das classes

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Class"
)

plt.title("Distribuição das Transações")
plt.xlabel("Classe (0 = Normal / 1 = Fraude)")
plt.ylabel("Quantidade")

plt.tight_layout()
plt.show()


# 6. CRIAÇÃO DA VARIÁVEL LOGAMOUNT

print("\n" + "=" * 60)
print("PREPARAÇÃO DOS DADOS")
print("=" * 60)

df["LogAmount"] = np.log1p(df["Amount"])

print("\nVariável LogAmount criada com sucesso.")


# 7. SEPARAÇÃO DAS VARIÁVEIS

# Time é retirado para evitar que o tempo absoluto da transação
# seja utilizado diretamente pelo modelo.

X = df.drop(
    columns=["Class", "Time"]
)

y = df["Class"]


print(f"\nQuantidade de variáveis utilizadas: {X.shape[1]}")


# 8. SEPARAÇÃO ENTRE TREINO E TESTE

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nDados separados:")
print(f"Treinamento: {len(X_train):,}")
print(f"Teste: {len(X_test):,}")


# 9. PADRONIZAÇÃO

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns,
    index=X_test.index
)


# 10. REGRESSÃO LOGÍSTICA

print("\n" + "=" * 60)
print("REGRESSÃO LOGÍSTICA")
print("=" * 60)

modelo_lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=RANDOM_STATE
)

modelo_lr.fit(
    X_train_scaled,
    y_train
)

y_pred_lr = modelo_lr.predict(X_test_scaled)

y_prob_lr = modelo_lr.predict_proba(
    X_test_scaled
)[:, 1]


print("\nRelatório de classificação:")
print(
    classification_report(
        y_test,
        y_pred_lr,
        target_names=["Normal", "Fraude"]
    )
)


precision_lr = precision_score(
    y_test,
    y_pred_lr
)

recall_lr = recall_score(
    y_test,
    y_pred_lr
)

f1_lr = f1_score(
    y_test,
    y_pred_lr
)

roc_auc_lr = roc_auc_score(
    y_test,
    y_prob_lr
)

pr_auc_lr = average_precision_score(
    y_test,
    y_prob_lr
)


# 11. RANDOM FOREST

print("\n" + "=" * 60)
print("RANDOM FOREST")
print("=" * 60)

modelo_rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

modelo_rf.fit(
    X_train,
    y_train
)

y_pred_rf = modelo_rf.predict(
    X_test
)

y_prob_rf = modelo_rf.predict_proba(
    X_test
)[:, 1]


print("\nRelatório de classificação:")

print(
    classification_report(
        y_test,
        y_pred_rf,
        target_names=["Normal", "Fraude"]
    )
)


precision_rf = precision_score(
    y_test,
    y_pred_rf
)

recall_rf = recall_score(
    y_test,
    y_pred_rf
)

f1_rf = f1_score(
    y_test,
    y_pred_rf
)

roc_auc_rf = roc_auc_score(
    y_test,
    y_prob_rf
)

pr_auc_rf = average_precision_score(
    y_test,
    y_prob_rf
)


# 12. IMPORTÂNCIA DAS VARIÁVEIS - RANDOM FOREST

importancias_rf = pd.Series(
    modelo_rf.feature_importances_,
    index=X_train.columns
).sort_values(
    ascending=False
)

print("\nPrincipais variáveis:")
print(importancias_rf.head(10))


plt.figure(figsize=(10, 7))

importancias_rf.head(15).sort_values().plot(
    kind="barh"
)

plt.title("Principais Variáveis - Random Forest")
plt.xlabel("Importância")

plt.tight_layout()
plt.show()


# 13. XGBOOST

print("\n" + "=" * 60)
print("XGBOOST")
print("=" * 60)

peso_classes = (
    y_train.value_counts()[0]
    / y_train.value_counts()[1]
)

print(
    f"\nPeso utilizado para a classe de fraude: "
    f"{peso_classes:.2f}"
)


modelo_xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=peso_classes,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

modelo_xgb.fit(
    X_train,
    y_train
)

y_prob_xgb = modelo_xgb.predict_proba(
    X_test
)[:, 1]

y_pred_xgb = (
    y_prob_xgb >= 0.5
).astype(int)


print("\nRelatório de classificação:")

print(
    classification_report(
        y_test,
        y_pred_xgb,
        target_names=["Normal", "Fraude"]
    )
)


precision_xgb = precision_score(
    y_test,
    y_pred_xgb
)

recall_xgb = recall_score(
    y_test,
    y_pred_xgb
)

f1_xgb = f1_score(
    y_test,
    y_pred_xgb
)

roc_auc_xgb = roc_auc_score(
    y_test,
    y_prob_xgb
)

pr_auc_xgb = average_precision_score(
    y_test,
    y_prob_xgb
)


# 14. COMPARAÇÃO DOS MODELOS

print("\n" + "=" * 60)
print("COMPARAÇÃO DOS MODELOS")
print("=" * 60)

resultados = pd.DataFrame({
    "Modelo": [
        "Regressão Logística",
        "Random Forest",
        "XGBoost"
    ],
    "Precisão": [
        precision_lr,
        precision_rf,
        precision_xgb
    ],
    "Recall": [
        recall_lr,
        recall_rf,
        recall_xgb
    ],
    "F1": [
        f1_lr,
        f1_rf,
        f1_xgb
    ],
    "ROC-AUC": [
        roc_auc_lr,
        roc_auc_rf,
        roc_auc_xgb
    ],
    "PR-AUC": [
        pr_auc_lr,
        pr_auc_rf,
        pr_auc_xgb
    ]
})

print(
    resultados.to_string(
        index=False
    )
)


# 15. CURVA ROC

fpr_lr, tpr_lr, _ = roc_curve(
    y_test,
    y_prob_lr
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    y_prob_rf
)

fpr_xgb, tpr_xgb, _ = roc_curve(
    y_test,
    y_prob_xgb
)


plt.figure(figsize=(10, 7))

plt.plot(
    fpr_lr,
    tpr_lr,
    label=f"Logística - AUC {roc_auc_lr:.3f}"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest - AUC {roc_auc_rf:.3f}"
)

plt.plot(
    fpr_xgb,
    tpr_xgb,
    label=f"XGBoost - AUC {roc_auc_xgb:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC")

plt.legend()

plt.tight_layout()
plt.show()


# 16. CURVA PRECISION-RECALL

precision_lr_curve, recall_lr_curve, _ = (
    precision_recall_curve(
        y_test,
        y_prob_lr
    )
)

precision_rf_curve, recall_rf_curve, _ = (
    precision_recall_curve(
        y_test,
        y_prob_rf
    )
)

precision_xgb_curve, recall_xgb_curve, _ = (
    precision_recall_curve(
        y_test,
        y_prob_xgb
    )
)


plt.figure(figsize=(10, 7))

plt.plot(
    recall_lr_curve,
    precision_lr_curve,
    label=f"Logística - AP {pr_auc_lr:.3f}"
)

plt.plot(
    recall_rf_curve,
    precision_rf_curve,
    label=f"Random Forest - AP {pr_auc_rf:.3f}"
)

plt.plot(
    recall_xgb_curve,
    precision_xgb_curve,
    label=f"XGBoost - AP {pr_auc_xgb:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precisão")
plt.title("Curva Precision-Recall")

plt.legend()

plt.tight_layout()
plt.show()


# 17. AJUSTE DO THRESHOLD

print("\n" + "=" * 60)
print("AJUSTE DO THRESHOLD")
print("=" * 60)

thresholds = np.arange(
    0.05,
    0.96,
    0.05
)

resultados_threshold = []


for threshold in thresholds:

    predicoes = (
        y_prob_xgb >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predicoes,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predicoes,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predicoes,
        zero_division=0
    )

    resultados_threshold.append({
        "Threshold": threshold,
        "Precisão": precision,
        "Recall": recall,
        "F1": f1
    })


threshold_df = pd.DataFrame(
    resultados_threshold
)

print("\nResultados:")
print(threshold_df.to_string(index=False))


# 18. GRÁFICO DO THRESHOLD

plt.figure(figsize=(10, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precisão"],
    marker="o",
    label="Precisão"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    marker="o",
    label="F1"
)

plt.xlabel("Threshold")
plt.ylabel("Métrica")
plt.title("Impacto do Threshold nas Métricas")

plt.legend()

plt.tight_layout()
plt.show()


# 19. ESCOLHA DO THRESHOLD

melhor_linha = threshold_df.loc[
    threshold_df["F1"].idxmax()
]

melhor_threshold = (
    melhor_linha["Threshold"]
)

print(
    f"\nThreshold selecionado: "
    f"{melhor_threshold:.2f}"
)

print(
    f"Precisão: "
    f"{melhor_linha['Precisão']:.4f}"
)

print(
    f"Recall: "
    f"{melhor_linha['Recall']:.4f}"
)

print(
    f"F1: "
    f"{melhor_linha['F1']:.4f}"
)


# 20. XGBOOST COM THRESHOLD AJUSTADO

y_pred_xgb_threshold = (
    y_prob_xgb >= melhor_threshold
).astype(int)


print("\nRelatório com threshold ajustado:")

print(
    classification_report(
        y_test,
        y_pred_xgb_threshold,
        target_names=["Normal", "Fraude"]
    )
)


precision_final = precision_score(
    y_test,
    y_pred_xgb_threshold
)

recall_final = recall_score(
    y_test,
    y_pred_xgb_threshold
)

f1_final = f1_score(
    y_test,
    y_pred_xgb_threshold
)


# 21. UNDERSAMPLING

print("\n" + "=" * 60)
print("UNDERSAMPLING")
print("=" * 60)

under_sampler = RandomUnderSampler(
    random_state=RANDOM_STATE
)

X_train_under, y_train_under = (
    under_sampler.fit_resample(
        X_train,
        y_train
    )
)

print("\nDistribuição após undersampling:")
print(
    pd.Series(
        y_train_under
    ).value_counts()
)


modelo_rf_under = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

modelo_rf_under.fit(
    X_train_under,
    y_train_under
)

y_prob_rf_under = (
    modelo_rf_under.predict_proba(
        X_test
    )[:, 1]
)

y_pred_rf_under = (
    y_prob_rf_under >= 0.5
).astype(int)


precision_under = precision_score(
    y_test,
    y_pred_rf_under
)

recall_under = recall_score(
    y_test,
    y_pred_rf_under
)

f1_under = f1_score(
    y_test,
    y_pred_rf_under
)


print(
    classification_report(
        y_test,
        y_pred_rf_under,
        target_names=["Normal", "Fraude"]
    )
)


# 22. OVERSAMPLING

print("\n" + "=" * 60)
print("OVERSAMPLING")
print("=" * 60)

over_sampler = RandomOverSampler(
    random_state=RANDOM_STATE
)

X_train_over, y_train_over = (
    over_sampler.fit_resample(
        X_train,
        y_train
    )
)

print("\nDistribuição após oversampling:")
print(
    pd.Series(
        y_train_over
    ).value_counts()
)


modelo_rf_over = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

modelo_rf_over.fit(
    X_train_over,
    y_train_over
)

y_prob_rf_over = (
    modelo_rf_over.predict_proba(
        X_test
    )[:, 1]
)

y_pred_rf_over = (
    y_prob_rf_over >= 0.5
).astype(int)


precision_over = precision_score(
    y_test,
    y_pred_rf_over
)

recall_over = recall_score(
    y_test,
    y_pred_rf_over
)

f1_over = f1_score(
    y_test,
    y_pred_rf_over
)


print(
    classification_report(
        y_test,
        y_pred_rf_over,
        target_names=["Normal", "Fraude"]
    )
)


# 23. COMPARAÇÃO DO BALANCEAMENTO

comparacao_balanceamento = pd.DataFrame({
    "Modelo": [
        "Random Forest",
        "Random Forest + Undersampling",
        "Random Forest + Oversampling"
    ],
    "Precisão": [
        precision_rf,
        precision_under,
        precision_over
    ],
    "Recall": [
        recall_rf,
        recall_under,
        recall_over
    ],
    "F1": [
        f1_rf,
        f1_under,
        f1_over
    ]
})


print("\n" + "=" * 60)
print("COMPARAÇÃO DO BALANCEAMENTO")
print("=" * 60)

print(
    comparacao_balanceamento.to_string(
        index=False
    )
)


# 24. SHAP

print("\n" + "=" * 60)
print("EXPLICABILIDADE COM SHAP")
print("=" * 60)

explainer = shap.TreeExplainer(
    modelo_xgb
)

shap_values = explainer.shap_values(
    X_test
)


# 25. IMPORTÂNCIA DAS VARIÁVEIS COM SHAP

print("\nGerando gráfico SHAP...")

shap.summary_plot(
    shap_values,
    X_test,
    show=True
)


# 26. EXPLICAÇÃO DE UMA TRANSAÇÃO

indice = 0

transacao = X_test.iloc[[indice]]

probabilidade_fraude = (
    modelo_xgb.predict_proba(
        transacao
    )[0][1]
)

print(
    f"\nProbabilidade de fraude "
    f"da transação analisada: "
    f"{probabilidade_fraude:.4f}"
)


shap_explanation = explainer(
    transacao
)

shap.plots.waterfall(
    shap_explanation[0]
)


# 27. MATRIZ DE CONFUSÃO DO MODELO FINAL

cm_final = confusion_matrix(
    y_test,
    y_pred_xgb_threshold
)


plt.figure(figsize=(7, 5))

sns.heatmap(
    cm_final,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Matriz de Confusão - XGBoost com Threshold Ajustado"
)

plt.xlabel("Previsão")
plt.ylabel("Real")

plt.tight_layout()
plt.show()


# 28. RESULTADO FINAL

resultado_final = pd.DataFrame({
    "Métrica": [
        "Threshold",
        "Precisão",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC"
    ],
    "Resultado": [
        melhor_threshold,
        precision_final,
        recall_final,
        f1_final,
        roc_auc_xgb,
        pr_auc_xgb
    ]
})


print("\n" + "=" * 60)
print("RESULTADO FINAL")
print("=" * 60)

print(
    resultado_final.to_string(
        index=False
    )
)


print("\nProjeto concluído!")
