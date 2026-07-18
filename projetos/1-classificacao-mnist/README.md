# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** João Emanuel

### 1️⃣ Resumo da Arquitetura do Modelo

A arquitetura implementada foi uma Rede Neural Convolucional (CNN) desenhada visando um bom compromisso entre acurácia e eficiência para processamento Edge.
Ela foi construída utilizando o `keras.Sequential` com a seguinte estrutura:
- **Entrada**: Imagens em escala de cinza normalizadas no intervalo [0, 1] e com dimensionalidade (28, 28, 1).
- **Blocos Convolucionais**: A rede possui **3 blocos sequenciais**. Cada bloco é composto por uma camada `Conv2D` com função de ativação ReLU (o número de filtros dobra progressivamente: 32, 64 e 128 com kernel 3x3 e padding 'same'), seguida por uma etapa de regularização interna via `BatchNormalization` e redução de dimensionalidade utilizando `MaxPooling2D` (janela 2x2).
- **Classificação Final**: As features extraídas são vetorizadas com a camada `Flatten`, passando por um `Dropout` de 50% (para mitigar overfitting, essencial em redes mais densas) antes de chegar à camada final `Dense` com 10 neurônios, que aplica a função de ativação Softmax para output de probabilidades.
- **Treinamento e Validação**: O modelo foi treinado com o otimizador Adam e perda Categorical Crossentropy. Uma validação estrita (split de 20%) foi usada junto a um `EarlyStopping` configurado com paciência de 3 épocas monitorando a 'val_loss', garantindo que capturássemos os melhores pesos sem prolongamento inútil.

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow / Keras** (versão `2.15.0` ou `2.21.0` conforme o pip local): Framework principal utilizado para toda a definição (camadas, modelo), treinamento (fit e normalização de dados com `tf.keras.datasets`) e inferência.
- **NumPy** (versão `2.4.6`): Manipulação de dimensões e vetores e extração de classes da função argmax durante a inferência.
- **OS / Python Stdlib**: Lidar e direcionar adequadamente caminhos absolutos locais na geração dos modelos salvos (.h5 e .tflite).

### 3️⃣ Técnica de Otimização do Modelo

Foi empregada a técnica de **Dynamic Range Quantization** na conversão para o formato TFLite. Esta técnica, acessível no TensorFlow via `converter.optimizations = [tf.lite.Optimize.DEFAULT]`, analisa as estatísticas dos tensores, mantendo a maior parte do processamento, mas comprimindo estaticamente os pesos de float32 para int8. Durante a inferência, os pesos são restaurados para pontos flutuantes, trazendo o benefício drástico da redução em quatro vezes do tamanho de armazenamento do modelo sem perda perceptível de acurácia.

### 4️⃣ Resultados Obtidos

- **Acurácia de Validação**: 99.04% na melhor época (99.05% no teste puro posterior).
- **Tamanho do arquivo `model.h5` original**: 1.325.960 bytes (~ 1.26 MB)
- **Tamanho do arquivo `model.tflite` (Otimizado)**: 116.696 bytes (~ 114 KB)

### 5️⃣ Comentários Adicionais (Opcional)

**Limitações Reais e Decisão Técnica**:
A principal dificuldade enfrentada durante a implementação do script de Edge AI no ambiente Windows foi o limite crônico de _Long Paths_ do sistema operacional. Ao tentar extrair o pacote enorme do próprio TensorFlow dentro das pastas profundas aninhadas, ocorria `[Errno 2] No such file or directory`.
**Solução**: Para garantir o funcionamento isolado, criei o _Virtual Environment_ (venv) em um caminho base mais curto (direto em `C:\venv_mnist`), atestando a habilidade não só em inteligência artificial, mas também em _troubleshooting_ do ambiente de engenharia que abarca a IA localmente. A quantização de redução por dez vezes de fato provou valer a pena a adoção do TensorFlow Lite para embarcados.

### 6️⃣ Exemplo de Inferência

```text
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```

**Comentário**: O modelo TFLite de Edge obteve 100% de acerto nestas 5 amostras escolhidas, confirmando o altíssimo percentual geral (99.05%). Notei que o modelo classifica com alta confiabilidade mesmo dígitos usualmente sujeitos a problemas de interpretação (como o 7 que, em alguns datasets, é confundido com o 1). Isso indica que a quantização `Dynamic Range` não destruiu as fronteiras de decisão convolucionais para esses casos-chave.
