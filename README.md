Detecção de Fraudes em Transações de Cartão

Projeto de Machine Learning desenvolvido para identificar possíveis transações fraudulentas em cartões de crédito.

O projeto aborda um dos principais desafios de problemas de detecção de fraude: o **forte desbalanceamento entre transações normais e fraudulentas**.

Objetivo

Desenvolver e comparar modelos de classificação capazes de identificar transações fraudulentas, utilizando métricas adequadas para um conjunto de dados altamente desbalanceado.

O projeto também utiliza técnicas de ajuste de threshold e explicabilidade com SHAP para entender as decisões do modelo.

---

Dataset

O dataset utilizado contém informações de transações de cartões de crédito.

Dataset:

https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

A base possui:

* `Time`
* `Amount`
* `V1` até `V28`
* `Class`

A variável `Class` representa o resultado da transação:

* `0` → transação normal
* `1` → transação fraudulenta

As variáveis `V1` até `V28` são componentes obtidos através de PCA e foram transformadas para preservar informações sensíveis das transações.

O dataset possui **284.807 transações**, sendo apenas **492 fraudes**, aproximadamente **0,17%** da base.

---

Desbalanceamento

O principal desafio do projeto é o desbalanceamento das classes.

Aproximadamente:

* 99,83% das transações são normais;
* 0,17% das transações são fraudulentas.

Por esse motivo, a acurácia não é suficiente para avaliar o desempenho dos modelos.

Um modelo que classificasse praticamente todas as transações como normais poderia apresentar uma acurácia muito alta e, ao mesmo tempo, praticamente não detectar fraudes.

Por isso, foram utilizadas principalmente:

* Precisão;
* Recall;
* F1-score;
* ROC-AUC;
* PR-AUC.

O **recall da classe fraude** recebe atenção especial porque representa a capacidade do modelo de identificar transações fraudulentas.

---

Tecnologias utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Imbalanced-learn
* SHAP
* Matplotlib
* Seaborn

---

Etapas do projeto

1. Exploração dos dados

O projeto começa carregando o dataset diretamente pela URL.

São analisados:

* quantidade de registros;
* quantidade de variáveis;
* tipos dos dados;
* estatísticas;
* valores nulos;
* distribuição das classes.

2. Preparação

Foi criada uma nova variável chamada `LogAmount`, utilizando uma transformação logarítmica sobre o valor das transações.

Também foi realizada a separação entre:

* variáveis preditoras (`X`);
* variável alvo (`y`).

Os dados foram divididos em treinamento e teste utilizando `stratify`, mantendo a proporção das classes.

A padronização foi realizada utilizando `StandardScaler`.

---

🤖 Modelos utilizados

Foram implementados três modelos principais:

Regressão Logística

Utilizada como modelo baseline.

Foi configurado `class_weight="balanced"` para considerar o desbalanceamento das classes.

Random Forest

Foi utilizado um Random Forest com pesos balanceados.

Além da classificação, suas importâncias de variáveis foram analisadas.

XGBoost

O XGBoost foi utilizado com `scale_pos_weight` para aumentar a importância da classe minoritária durante o treinamento.

---

Técnicas de balanceamento

Também foram testadas duas estratégias:

Undersampling

Redução da quantidade de exemplos da classe majoritária.

Oversampling

Aumento da quantidade de exemplos da classe minoritária.

Os resultados foram comparados através de precisão, recall e F1-score.

---

Ajuste do Threshold

Os modelos de classificação normalmente utilizam um threshold de `0.5`.

Neste projeto foram testados diversos thresholds entre `0.05` e `0.95`.

O objetivo foi observar como a alteração do threshold influencia:

* precisão;
* recall;
* F1-score.

O threshold utilizado no resultado final foi selecionado com base no maior F1-score encontrado durante os testes.

---

Avaliação

Foram utilizadas:

Matriz de Confusão

Permite visualizar:

* verdadeiros positivos;
* verdadeiros negativos;
* falsos positivos;
* falsos negativos.

Curva ROC

Utilizada para analisar a capacidade de separação entre as classes.

Curva Precision-Recall

É especialmente relevante neste projeto devido à grande diferença entre a quantidade de transações normais e fraudulentas.

---

Explicabilidade com SHAP

O projeto utiliza SHAP para compreender as decisões do modelo XGBoost.

Foram realizadas duas análises:

Importância global

Identifica quais variáveis tiveram maior influência nas previsões do modelo.

Explicação individual

Uma transação específica é analisada para identificar quais características contribuíram para aumentar ou diminuir sua probabilidade prevista de fraude.

---

Estrutura do projeto

```text
fraude-cartao/
│
├── README.md
├── requirements.txt
└── fraude_cartao.py
```

---

Como executar

1. Clone o repositório

```bash
git clone URL_DO_SEU_REPOSITORIO
```

2. Entre na pasta

```bash
cd fraude-cartao
```

3. Instale as dependências

```bash
pip install -r requirements.txt
```

4. Execute o projeto

```bash
python fraude_cartao.py
```

O programa fará o download do dataset diretamente pela URL e executará as etapas de análise, treinamento, avaliação, balanceamento e explicabilidade.

---

Resultados

Os resultados das métricas são exibidos diretamente durante a execução do programa.

As principais métricas apresentadas são:

| Modelo              | Precisão |   Recall |       F1 |  ROC-AUC |   PR-AUC |
| ------------------- | -------: | -------: | -------: | -------: | -------: |
| Regressão Logística | Executar | Executar | Executar | Executar | Executar |
| Random Forest       | Executar | Executar | Executar | Executar | Executar |
| XGBoost             | Executar | Executar | Executar | Executar | Executar |

Os resultados podem variar conforme a configuração do ambiente e dos modelos.

---

Possíveis melhorias

Como evolução do projeto, podem ser implementadas:

* GridSearchCV para busca de hiperparâmetros;
* novos algoritmos de classificação;
* criação de variáveis comportamentais;
* análise de transações em sequência;
* ajuste de threshold baseado em custos de falsos positivos e falsos negativos;
* criação de uma API para disponibilizar o modelo;
* desenvolvimento de uma interface para análise de transações.

---

Fonte

Dataset disponibilizado pelo TensorFlow:

https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv

Projeto desenvolvido para estudo de Machine Learning, classificação desbalanceada e explicabilidade de modelos.
