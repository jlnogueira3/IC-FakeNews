# Feedback ponderado com validação temporal

Implementação experimental, em arquivo separado, para a IC de detecção de fake news em português. **Não altera notebooks originais.**

## Protocolo científico

- **Inicialização:** toda a Fake.br, convenção `true=0`, `fake=1`.
- **Fluxo de teste:** FakeRecogna, datas em ordem crescente, a partir de 2020-01-01 (configurável). Na FakeRecogna original, `Classe=0` indica **fake** e `Classe=1` indica **true**; a conversão é realizada na leitura.
- **Representação:** `HashingVectorizer` (1-2 gramas, 2**18 features), com intercepto explícito.
- **Classificador:** `SGDClassifier(loss="log_loss", penalty=None, fit_intercept=False, learning_rate="constant")`.
- **Avaliação prequential:** *todas* as notícias do mesmo dia são previstas antes de qualquer atualização desse dia; previsões e rótulos são armazenados para F1 macro mensal.
- **Separação treinamento/validação:** cada quinta notícia cronológica fica reservada exclusivamente para a validação temporal (20% determinísticos). Não é utilizada no treinamento dos modelos ponderado e peso-constante, inclusive depois que sair da janela.
- **Referência temporal:** os últimos 90 dias de notícias reservadas **de datas estritamente anteriores** ao dia avaliado, até 256 exemplos. Se houver menos de 20 exemplos ou faltar alguma classe, o peso permanece em 0,5 (cold start).
- **Disponibilização do rótulo:** o protocolo simula feedback verdadeiro imediato ao final da predição. Referências do dia só ingressam na validação após o processamento inteiro desse dia.
- **Meta-gradiente:** cada exemplo não reservado recebe peso `clip(0.5 - alpha_meta * dL_val/dw, 0.05, 1)`, com gradiente calculado no passo virtual sobre a validação recente. O `partial_fit` aplica o peso; o estimador não calcula pesos sozinho.
- **Baselines comparáveis ao ponderado:** (i) `imediato_peso_constante` utiliza o mesmo subconjunto de treino e `sample_weight=0.5`; (ii) `imediato_todas_noticias` aprende com todas as notícias, inclusive reservadas, e usa peso 1. Ambos usam o **mesmo SGD com perda logística** do modelo ponderado.
- **Baselines históricos reavaliados sem vazamento:** `offline_{Perceptron,PassiveAggressiveClassifier,MultinomialNB,SGDClassifier}` (estáticos) e `imediato_{...}` (atualizados após a predição do dia) usam os quatro classificadores com configurações padrão do notebook original. **O SGDClassifier original usa hinge e não deve ser confundido com o SGD logístico ponderado.**

O meta-gradiente binário para regressão logística é:

\[
g_i=(\sigma(\theta^\top x_i)-y_i)x_i, \quad
\hat\theta=\theta-\eta w_i g_i, \quad
\frac{\partial L_{val}}{\partial w_i}=-\eta\,\nabla_\theta L_{val}(\hat\theta)^\top g_i.
\]

O intercepto é a última dimensão constante do vetor `x_i`. Com `penalty=None`, este é o gradiente exato do passo virtual definido acima (erro de arredondamento à parte). A taxa `eta` define a escala da atualização dos coeficientes; `alpha_meta` define a escala do peso.

**Atenção metodológica:** esta é uma *adaptação online de um passo* da ideia de meta-reweighting de Zhang et al. (2025), com conjunto de referência temporal, e **não** a reprodução do FBR/NTK original. Cada notícia nova tem peso inicial 0,5 e recebe uma atualização de peso antes do `partial_fit`. O piso 0,05 é uma escolha de regularização prática; o artigo teoricamente permite pesos 0.

## Executar na máquina local

Dentro de `/home/joao-lucas-nogueira/Área de trabalho/Projetos/IC/Codigos`, atualize o repositório e traga a branch do PR, ou faça checkout após o merge.

Os arquivos esperados no diretório da IC:

- `database/Fake.br/corpus/full_texts/{true,fake}/*.txt`;
- `database/FakeRecogna/repositorio/dataset/FakeRecogna_datas_padronizadas.xlsx` **ou** `database/FakeRecogna/FakeRecogna.xlsx` (colunas `Noticia`, `Classe`, `Data`; título e subtítulo opcionais).

Instalar dependências:

```bash
python -m pip install numpy scipy pandas scikit-learn openpyxl pytest
```

Rodar da raiz do projeto:

```bash
python code/feedback_ponderado_temporal/feedback_ponderado_temporal.py \
  --janela-dias 90 --reservar-cada 5 --min-validacao 20 \
  --max-validacao 256 --eta 0.05 --alpha-meta 10.0
```

Testes de unidade e integridade temporal:

```bash
python -m pytest code/feedback_ponderado_temporal/tests -q
```

Os resultados são salvos em `result/feedback_ponderado_temporal/`:

- `predicoes_e_pesos.csv`: data, índice, rótulo, previsões antes do treino, seleção para validação, tamanho da referência, meta-gradiente e peso utilizado.
- `metricas_mensais.csv`: `accuracy`, `precision_macro`, `recall_macro` e `f1_macro` para os 11 cenários, com zero_division=0.

## Limitações e próximos experimentos

1. O classificador ponderado recebe menos notícias para treinamento do que o baseline ideal; compare principalmente com `imediato_peso_constante`, que utiliza **a mesma seleção**, para isolar o efeito da ponderação.
2. Os hiperparâmetros 90 dias/20%/256 notícias/eta/alpha são escolhas iniciais, não parâmetros validados empiricamente. Avaliar sensibilidade e repetir com diferentes seeds e estratégias de reserva.
3. Quando a base de validação é pequena, desbalanceada ou defasada, o meta-gradiente pode não refletir novos padrões do concept drift. A notícia mais recente só passa a compor a referência em dias posteriores.
4. Se houver duplicatas ou quase duplicatas entre as duas bases, separar/identificar esses registros antes de afirmar ganho de generalização.
5. Não foi introduzido ruído artificial nos rótulos; resultados futuros devem diferenciar *noise robustness* de adaptação a *concept drift*.
6. Não executar seleção de hiperparâmetros ou calibração usando notícias futuras; nenhuma métrica exibida aqui é resultado experimental obtido nas bases reais.

## Por que os gráficos anteriores de feedback imediato podem estar inflados?

No notebook `code/feedback_imediato/treinamento_offline.ipynb`, a célula que gera as curvas usa `Perceptron.treinar(i)` **antes** de `Perceptron.f1macro(i)`, repetindo isso para outros modelos. Em `treinar_mes_fakerecogna`, `partial_fit` recebe as mesmas notícias do mês que serão avaliadas em `f1macro(i)`. Assim, a métrica obtida é pós-treinamento no próprio mês, não uma estimativa prequential real. Não usar os valores históricos como baseline causal.

Esta implementação corrige a comparação ao obter **todas as predições antes do feedback do dia**, tanto para estático quanto para imediato e ponderado. Os gráficos anteriores incluem novembro e dezembro de 2019, cujas pontuações não estão necessariamente no fluxo FakeRecogna pós-corte; a comparação aqui começa em janeiro de 2020. O SGD padrão do notebook (hinge) também é diferente do modelo experimental (log_loss).

Avaliá-los lado a lado **não prova** que o feedback ponderado deva superar os melhores modelos. A pergunta experimental correta é se ele melhora, sob condições iguais, o desempenho do **mesmo classificador** em relação ao feedback imediato.
