# Organização dos experimentos de fake news

## Estrutura

- `code/feedback_imediato/treinamento_offline.ipynb`: parte de feedback imediato do notebook offline original.
- `code/feedback_incerto/treinamento_feedback_incerto.ipynb`: parte de feedback incerto do mesmo notebook, com suas células comuns.
- `code/feedback_mensal/treinamento_online.ipynb`: treinamento online por mês.
- `code/treinamento_estatico/`: notebook de treinamento estático e testes exploratórios que já estavam nele.
- `code/testes_faketweet/`: classificador de teste do FakeTweet, preservado como experimento exploratório; esta base não foi utilizada nos experimentos principais.
- `code/preparacao/`: preparação da divisão temporal do Fake.br.
- `code/visualizacao/`: funções compartilhadas para gráficos.
- `code/estudos/`: notebooks didáticos de representação tradicional e embeddings.
- `code/apoio/`: anotações e recursos dos estudos (PorSimples e embeddings).
- `result/`: gráficos, tabelas e saídas históricas separados pelos mesmos tipos de experimento.
- `database/FakeRecogna/`: planilha usada pelos notebooks offline/estático e repositório com a versão de datas padronizadas usada pelo notebook online.
- `database/Fake.br/corpus/`: corpus antes localizado em `dados/Fake.br-Corpus-master`.
- `database/Fake.br/corpus_copia_original/`: outra cópia, antes em `Fake.br-Corpus`, mantida com seu Git e utilizada pelo script de divisão temporal.
- `database/Fake.br/processados/`: dataframes existentes e divisão temporal existente.
- `database/FakeTweet/`: base FakeTweetBr.csv, README e histórico Git originais.

## Uso e resultados

Abra os notebooks com o diretório de trabalho em `Codigos` ou em uma subpasta. A primeira célula encontra a raiz automaticamente. Os caminhos de entrada apontam para `database/`; as novas saídas apontam para `result/<tipo>/`. Dataframes derivados de cada execução ficam em `result/<tipo>/processados/`, evitando sobrescrever os dataframes históricos.

O notebook offline original reúne feedback imediato e incerto; foi separado sem modificar as regras dos modelos. A sequência original das células foi preservada. Os notebooks são experimentos existentes e podem depender de execução interativa prévia; esta organização não constitui validação de execução do início ao fim.

As saídas já salvas nas células foram exportadas como JSON e texto; imagens incorporadas foram extraídas como PNG. Não foram recalculadas métricas. Gráficos e pickles existentes foram movidos sem alteração. O gráfico `grafico_resultados_feedback.png` pertence à seção de feedback incerto. Os artefatos mensais estão juntos em `result/feedback_mensal/`; o pickle mensal preexistente não tem uma célula geradora identificada nos scripts encontrados.

Os dois pontos de novembro/dezembro de 2019 do gráfico de feedback de 2020 são valores digitalizados de uma figura, conforme o comentário original no notebook. Não são novas medições.

## Preservação e pendências

`result/organizacao/scripts_originais.zip` contém os scripts e notebooks originais, inclusive o offline completo. `manifesto_movimentacoes.json` registra origem, destino e hashes de todos os arquivos inventariados; `verificacao.txt` informa a conferência e pendências de sintaxe encontradas nos notebooks originais.

O classificador de teste referencia `merge-val12-8.csv`, ausente no material original. O estudo de embeddings também possui um caminho de exemplo para GoogleNews que precisa ser configurado se essa seção for usada. Há dependências de estado e erros já existentes nos notebooks; os algoritmos não foram revisados nesta reorganização.

Os diretórios ocultos `.venv` e `.idea` foram mantidos no lugar para preservar o ambiente e as configurações. Os repositórios das bases conservam `.git`; mover os arquivos de experimentos para fora do FakeTweet aparece como remoção no estado local desse repositório, sem criar commits.
