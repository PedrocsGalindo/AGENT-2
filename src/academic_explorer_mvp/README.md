# Academic Explorer MVP

## To do 
- enrich initial query 
- review the number os papers the model get
- review with information of the paper the model recives 

## Arquitetura

```text
src/academic_explorer_mvp/
|-- main.py
|-- config.py
|-- domain/
|-- providers/
|-- services/
|-- llm/
|-- graph/
`-- README.md
```

- `main.py`: CLI e resumo no terminal.
- `config.py`: variaveis de ambiente e configuracao pequena.
- `domain/`: dataclasses e estado tipado.
- `providers/`: chamadas diretas para OpenAlex e Semantic Scholar.
- `services/`: busca, normalizacao, deduplicacao, ranking e planejamento.
- `llm/`: modelo local via transformers e prompts.
- `graph/`: nos, roteamento e montagem do LangGraph.

## Fluxo do grafo

```text
initialize_context
  -> plan_filters
  -> plan_queries
  -> search_papers
  -> normalize_papers
  -> deduplicate_papers
  -> validate_papers
  -> judge_paper_validations
  -> decide_next_step
  -> ask_paper_feedback
      -> sim: finalize
      -> critica: analyze_search_feedback -> plan_queries
```

## Como a busca funciona

O usuario informa uma query inicial, por exemplo: "deteccao de violencia em audio".
O sistema primeiro avalia se essa query tem contexto suficiente para comecar uma
busca academica util. Se faltar contexto, ele faz uma pergunta de clarificacao
para entender melhor a intencao da pesquisa.

Depois da query estar clara, o sistema gera de 1 a 3 queries academicas que serao
usadas para buscar artigos. Antes de pesquisar, essas queries sao exibidas no
terminal. O parametro `--limit` controla quantos resultados sao buscados por
query/rodada, conforme a implementacao atual. O parametro `--max-rounds`
controla quantas rodadas de busca podem acontecer.

Depois de cada rodada, os artigos encontrados passam por uma validacao semantica.
Essa validacao tenta separar artigos realmente relacionados ao tema daqueles que
so tem palavras parecidas. Se o usuario criticar os resultados, o sistema usa
esse feedback para ajustar a direcao da busca nas proximas rodadas.

Exemplo:

```powershell
py -m academic_explorer_mvp.main --query "detecção de violência em áudio" --min-year 2020 --max-rounds 3 --limit 5
```

- `--query`: tema inicial da busca.
- `--min-year`: ano minimo dos artigos.
- `--max-rounds`: numero maximo de rodadas de busca.
- `--limit`: quantidade de resultados buscados por rodada/query, conforme a implementacao.

## Ambiente

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip setuptools wheel certifi

python -m pip install --use-feature=truststore --upgrade pip setuptools wheel certifi
python -m pip install --use-feature=truststore -e ".[local-model]"

py -m compileall src\academic_explorer_mvp
py -m academic_explorer_mvp.main --query "detecção de violencia em audio" --min-year 2020 --max-rounds 1 --limit 5
```

### Execucao validada

O fluxo completo roda com LangGraph, modelo local, OpenAlex, normalizacao,
deduplicacao, ranking e `stop_reason`. O Semantic Scholar pode responder HTTP
429 sem chave/API quota; nesse caso o erro aparece no resumo e o fluxo continua
com os resultados disponiveis dos outros providers.

## Variaveis de ambiente (.env)
Para uma validacao inicial, use um modelo pequeno:
```bash
OPENALEX_MAILTO =
SEMANTIC_SCHOLAR_API_KEY =
ACADEMIC_EXPLORER_PROVIDER_ = 
ACADEMIC_EXPLORER_AGENT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ACADEMIC_EXPLORER_AGENT_MAX_NEW_TOKENS = "512"
ACADEMIC_EXPLORER_AGENT_TEMPERATURE = "0.0"
ACADEMIC_EXPLORER_AGENT_TORCH_DTYPE = "auto"
ACADEMIC_EXPLORER_AGENT_DEVICE_MAP ="auto"
```

## Modelo local obrigatorio

O `QueryPlanner` e a unica camada que conversa com o modelo. Ele usa o modelo
para queries iniciais, refinamento e decisao de continuidade. Se `transformers`,
`torch`, `accelerate`, o modelo configurado, memoria, device map ou cache local
nao estiverem corretos, o CLI para com mensagem didatica.


## Limitacoes atuais

- Sem banco de dados.
- Sem vector store.
- Sem parsing de PDF.
- Sem frontend web.
- Sem feedback humano.
- Deduplicacao sem fuzzy matching.
- Ranking simples, deterministico e explicavel.

## Proximos passos

- Adicionar etapa `ask_feedback` e `apply_feedback`.
- Persistir resultados importantes.
- Expandir testes de grafo com providers falsos.
- Avaliar modelos locais menores ou mais rapidos.
- Explorar criterios de parada mais ricos.

## Test - Benchmark

| Categoria | Query | Resultado esperado |
|---|---|---|
| Clara | `violence detection in audio using deep learning` | Buscar direto |
| Clara | `dengue outbreak prediction using climate variables and machine learning` | Buscar direto |
| Clara | `stock price prediction using LSTM models` | Buscar direto |
| Pouca informação | `violence detection` | Pedir clarificação |
| Pouca informação | `dengue` | Pedir clarificação |
| Pouca informação | `AI in health` | Pedir clarificação |
| Mal formulada | `audio violence ai detect` | Normalizar intenção |
| Mal formulada | `dengue predict weather machine` | Normalizar intenção |
| Erro de digitação | `violnce detction in audoi` | Corrigir intenção |
| Erro de digitação | `medcal imag clasification` | Corrigir intenção |
| Ambígua | `bias detection` | Pedir clarificação |
| Ambígua | `emotion recognition` | Pedir clarificação |
| Ampla demais | `machine learning` | Pedir clarificação |
| Ampla demais | `computer vision` | Pedir clarificação |
| Específica | `sensor fusion for autonomous vehicle localization using IMU GNSS and Kalman filter` | Buscar direto |
| Específica | `credit card fraud detection using isolation forest and imbalanced datasets` | Buscar direto |
| Risco de drift | `AI for shopping` | Clarificar ou filtrar bem |
| Risco de drift | `audio classification` | Pedir clarificação |
| Português | `detecção de violência em áudio` | Buscar direto |
| Português | `previsão de dengue usando clima` | Buscar direto |
| PT/EN | `stock prediction usando LSTM` | Normalizar intenção |
| PT/EN | `fake news detection em redes sociais` | Normalizar intenção |
| Ruim realista | `quero artigos sobre ia` | Pedir clarificação |
| Ruim realista | `modelo que prevê coisa` | Pedir clarificação |
| Ranking | `fake news detection using NLP` | Testar ordenação dos melhores papers |
| Ranking | `medical image classification using CNN` | Testar ranking por relevância |
