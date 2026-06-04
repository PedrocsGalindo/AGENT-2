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
  -> plan_queries
  -> search_papers
  -> normalize_papers
  -> deduplicate_papers
  -> rank_papers
  -> decide_next_step
      -> continue: plan_queries
      -> finalize: finalize
```

O feedback humano nao foi implementado. O ponto previsto para depois e:

```text
rank_papers -> ask_feedback -> apply_feedback -> decide_next_step
```

## Ambiente

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip setuptools wheel certifi

python -m pip install --use-feature=truststore --upgrade pip setuptools wheel certifi
python -m pip install --use-feature=truststore -e ".[local-model]"

py -m compileall src\academic_explorer_mvp
py -m academic_explorer_mvp.main --query "audio violence detection" --min-year 2020 --max-rounds 1 --limit 5
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
