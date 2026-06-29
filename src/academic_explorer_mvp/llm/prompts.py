from dataclasses import dataclass, field

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.paper import Paper


@dataclass(frozen=True)
class PromptSpec:
    """Prompt versionado usado pelo modelo local."""

    name: str
    version: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


INITIAL_QUERIES_PROMPT_VERSION = "1.1.0"
PLAN_FILTERS_PROMPT_VERSION = "1.1.0"
REFINE_QUERIES_PROMPT_VERSION = "1.1.0"
FEEDBACK_ANALYSIS_PROMPT_VERSION = "1.1.0"
VALIDATE_PAPERS_PROMPT_VERSION = "1.1.0"
JUDGE_PAPER_VALIDATIONS_PROMPT_VERSION = "1.0.0"
CONTINUE_DECISION_PROMPT_VERSION = "1.0.1"
ENRICH_QUERY_PROMPT_VERSION = "1.0.1"
ASSESS_QUERY_CONTEXT_PROMPT_VERSION = "1.1.1"
CONTEXT_QUESTION_PROMPT_VERSION = "1.1.0"
REWRITE_USER_QUERY_PROMPT_VERSION = "1.1.0"
REWRITE_FROM_USER_REVISION_PROMPT_VERSION = "1.1.0"


def build_assess_query_context_prompt(context: SearchContext) -> PromptSpec:
    """Prompt para decidir se a query tem contexto suficiente."""

    text = f"""Retorne somente um objeto JSON. Sem markdown. Sem explicações fora do JSON.

Você está preparando uma busca de artigos acadêmicos.

Tarefa:
Decida se a consulta do usuário tem contexto suficiente para começar a criar queries acadêmicas úteis.

Regra de idioma:
- O campo reason deve ser em português para facilitar depuração.
- Mantenha as chaves JSON em inglês.

Critério central:
A consulta precisa conter:
1. um tema acadêmico reconhecível; e
2. intenção suficiente para evitar adivinhar a direção desejada.

Ser pesquisável não é o mesmo que ter contexto suficiente.
Se for possível criar uma busca geral, mas o significado ou a direção desejada ainda forem ambíguos, falta contexto.

Use has_enough_context=true quando:
- o usuário pedir explicitamente uma busca geral, visão geral, introdução, tendências, review, survey ou estado da arte;
- ou houver tema claro e foco suficiente, como método, dataset, fonte de dados, métrica, comparação, restrição, aplicação, domínio ou problema específico;
- e for possível gerar queries acadêmicas sem assumir o significado pretendido.

Use has_enough_context=false quando:
- a consulta for ampla e não indicar direção geral nem foco específico;
- houver termo ambíguo capaz de gerar buscas acadêmicas diferentes;
- o tema puder significar métodos, domínios, modalidades, sistemas, fontes de dados ou aplicações diferentes;
- gerar queries exigir assumir a intenção do usuário.

Regra de ambiguidade:
Se um termo central for ambíguo, retorne has_enough_context=false e explique a ambiguidade em reason.
Termos como agent, bias, automation, shopping automation, noise, detection, security, prediction e models podem ser ambíguos quando o sentido não estiver claro.

Regra de intenção geral:
Se a consulta disser "general", "overview", "visão geral", "geral", "quero entender", "trend", "trends", "state of the art", "survey" ou "review", considere que há intenção geral suficiente.
Não force review/survey como único tipo de resultado; apenas reconheça que o usuário deseja busca ampla.

Exemplos:

Consulta: stock prediction
Saída:
{{"has_enough_context":false,"reason":"A consulta é pesquisável, mas não informa se o usuário quer uma busca geral ou um foco como métodos, fontes de dados, ativos, mercados, horizonte temporal ou métricas."}}

Consulta: stock prediction geral
Saída:
{{"has_enough_context":true,"reason":"O usuário indicou explicitamente uma direção geral e ampla para o tema."}}

Consulta: stock price prediction using LSTM and news sentiment
Saída:
{{"has_enough_context":true,"reason":"A consulta especifica tarefa, método e fonte de dados."}}

Consulta: automated shopping
Saída:
{{"has_enough_context":false,"reason":"O termo automated shopping é ambíguo e pode indicar shopping agents, automação de e-commerce, checkout automático, sistemas de recomendação ou operações de varejo."}}

Consulta: audio violence detection
Saída:
{{"has_enough_context":false,"reason":"A consulta especifica tarefa e modalidade, mas não indica se o usuário deseja uma direção geral ou um foco como modelos, datasets, características, métricas ou implantação."}}

Consulta: audio violence detection geral
Saída:
{{"has_enough_context":true,"reason":"A consulta especifica detecção de violência por áudio e indica intenção geral."}}

Consulta: noisy data in medical image classification
Saída:
{{"has_enough_context":false,"reason":"A consulta é relevante, mas noisy data pode significar ruído de rótulo, ruído de imagem, ruído de aquisição, outliers ou robustez."}}

Consulta: label noise in medical image classification
Saída:
{{"has_enough_context":true,"reason":"A consulta especifica o tipo de ruído e o domínio de aplicação."}}

Consulta atual:
{context.user_query}

Formato JSON obrigatório:
{{"has_enough_context":true,"reason":"justificativa curta em português"}}
ou
{{"has_enough_context":false,"reason":"justificativa curta em português"}}
"""

    return PromptSpec(
        name="assess_query_context",
        version=ASSESS_QUERY_CONTEXT_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "query_context_assessment"},
    )


def build_context_question_prompt(user_query: str, reason: str) -> PromptSpec:
    """Prompt para gerar uma pergunta de esclarecimento."""

    text = f"""Retorne somente um objeto JSON. Sem markdown. Sem explicações fora do JSON.

Você está preparando uma busca de artigos acadêmicos.

Tarefa:
Gere exatamente uma pergunta útil de esclarecimento para a consulta atual do usuário.

Importante:
- Não reavalie se a consulta tem contexto suficiente; essa avaliação já foi feita.
- Faça somente uma pergunta.
- A pergunta deve ajudar a identificar a direção acadêmica pretendida.

Consulta atual:
{user_query}

Justificativa da avaliação anterior:
{reason}

Regra de idioma:
- question deve ser em português porque será mostrado ao usuário.
- reason deve ser em português para facilitar depuração.
- Mantenha as chaves JSON em inglês.

Regras:
- Construa a pergunta com base na consulta real e no contexto ausente descrito em reason.
- Se houver ambiguidade, mencione a ambiguidade e ofereça significados prováveis.
- Se houver amplitude excessiva, pergunte se o usuário deseja direção geral ou foco específico.
- Não copie exemplos literalmente.
- Não faça perguntas genéricas como "o que você quer saber?".
- Não adicione opções não relacionadas.
- Não pergunte sobre uma modalidade se ela já estiver clara; pergunte sobre o foco.

Dimensões úteis quando relevantes:
direção geral, método, família de modelos, aplicação, domínio, modalidade, dataset, fonte de dados, métrica, comparação, população, doença, tarefa, tipo de sistema, implantação, avaliação ou foco teórico/prático.

Nunca pergunte:
- Qual aspecto específico você precisa?
- Pode fornecer mais contexto?
- O que você quer saber?
- Esclareça sua consulta.
- Pode ser mais específico?

Orientação por tipo de tema:
- Para prediction/forecasting: métodos, fontes de dados, variável-alvo, horizonte temporal, mercado/domínio, métricas ou direção geral.
- Para detection/classification: modalidade, tipo de dado, datasets, modelos, extração de características, tempo real, benchmarks, métricas ou direção geral.
- Para agents: diferencie LLM/VLM agents, software agents, sistemas autônomos, recommendation agents, shopping agents, robótica ou workflows.
- Para bias: diferencie viés estatístico, viés de dataset, shortcut bias, fairness bias, viés de modelo, viés médico, viés social ou viés de avaliação.
- Para noise: diferencie label noise, input noise, ruído de aquisição, outliers, ambientes ruidosos ou robustez.

Exemplos:

Consulta: automated shopping
Saída:
{{"question":"O termo \"automated shopping\" ficou ambíguo: você quer falar de agentes de compra com IA, automação em e-commerce, checkout automático, sistemas de recomendação ou operações de varejo?","reason":"A pergunta expõe a ambiguidade e oferece interpretações acadêmicas prováveis."}}

Consulta: audio violence detection
Saída:
{{"question":"Você quer uma visão geral sobre detecção de violência por áudio ou quer focar em algo mais específico, como modelos, datasets, extração de características, métricas ou detecção em tempo real?","reason":"A pergunta preserva a modalidade de áudio e solicita o foco de pesquisa que ainda está ausente."}}

Consulta: noisy data in medical image classification
Saída:
{{"question":"Em dados ruidosos para classificação de imagens médicas, você quer focar em rótulos ruidosos, ruído na imagem, ruído de aquisição, outliers, robustez do modelo ou uma visão geral?","reason":"A pergunta diferencia os sentidos técnicos possíveis de dados ruidosos no tema do usuário."}}

Formato JSON obrigatório:
{{"question":"uma pergunta de esclarecimento com opções específicas do tema","reason":"justificativa curta em português"}}
"""

    return PromptSpec(
        name="context_question",
        version=CONTEXT_QUESTION_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "query_context_question"},
    )


def build_rewrite_user_query_prompt(
    initial_query: str,
    question: str,
    question_reason: str,
    user_answer: str,
) -> PromptSpec:
    """Prompt para propor um tema refinado após esclarecimento."""

    text = f"""Retorne somente um objeto JSON. Sem markdown. Sem explicações fora do JSON.

Você está preparando uma busca de artigos acadêmicos.

Tarefa:
Crie um único tema acadêmico mais claro combinando a consulta inicial, a pergunta de esclarecimento e a resposta do usuário.
Esta saída não é uma lista de queries; é um tema refinado que será usado no planejamento e na validação semântica.

Regra de idioma:
- proposed_query deve permanecer em inglês porque será usada na busca acadêmica.
- message deve ser em português porque será mostrado ao usuário.
- reason deve ser em português para facilitar depuração.
- Mantenha as chaves JSON em inglês.

Regras centrais:
- Crie exatamente um tema acadêmico refinado.
- Preserve o tema principal e os termos técnicos úteis da consulta inicial.
- Use a pergunta apenas para interpretar a resposta do usuário.
- Traduza o significado pretendido para inglês em proposed_query.
- Use somente informações fornecidas pelo usuário.
- Não invente métodos, datasets, métricas, domínios, aplicações, modalidades ou restrições.
- Não crie múltiplas queries.
- Não deixe o tema mais específico do que a resposta permite.
- Se a resposta adicionar um foco, inclua esse foco.
- Se a resposta for geral, mantenha o tema amplo.
- Se o usuário selecionar uma opção, inclua somente essa opção.
- Se o usuário rejeitar uma opção, não a inclua.

Resposta geral:
Se o usuário disser "geral", "visão geral", "quero entender", "panorama", "algo amplo", "não sei ainda" ou equivalente:
- mantenha o tema amplo;
- não transforme automaticamente em review, survey ou systematic review;
- não adicione método, dataset, métrica ou restrição específica.

Review/survey:
Use "review", "survey", "systematic review", "literature review" ou "state of the art" em proposed_query somente se o usuário pedir explicitamente.
Uma resposta geral não significa pedido exclusivo por artigos de revisão.

Exemplos:

Consulta inicial: audio violence detection
Resposta: geral
Saída:
{{"proposed_query":"general audio-based violence detection","message":"Com base na sua resposta, a busca ficaria: general audio-based violence detection","reason":"O usuário pediu uma direção ampla, preservando o tema de detecção de violência baseada em áudio."}}

Consulta inicial: audio violence detection
Resposta: revisão sistemática
Saída:
{{"proposed_query":"systematic review of audio-based violence detection","message":"Com base na sua resposta, a busca ficaria: systematic review of audio-based violence detection","reason":"O usuário pediu explicitamente uma revisão sistemática."}}

Consulta inicial: automated shopping
Resposta: agentes de compra com LLM e VLM
Saída:
{{"proposed_query":"shopping agents using LLMs and VLMs","message":"Com base na sua resposta, a busca ficaria: shopping agents using LLMs and VLMs","reason":"O usuário esclareceu que automated shopping significa agentes de compra com LLMs e VLMs."}}

Consulta inicial: shortcut bias
Resposta: imagens médicas
Saída:
{{"proposed_query":"shortcut bias in medical imaging","message":"Com base na sua resposta, a busca ficaria: shortcut bias in medical imaging","reason":"O usuário selecionou medical imaging como contexto de aplicação."}}

Consulta inicial:
{initial_query}

Pergunta de esclarecimento:
{question}

Justificativa da pergunta:
{question_reason}

Resposta do usuário:
{user_answer}

Formato JSON obrigatório:
{{"proposed_query":"tema acadêmico claro em inglês","message":"Com base na sua resposta, a busca ficaria: ...","reason":"justificativa curta em português"}}
"""

    return PromptSpec(
        name="rewrite_user_query",
        version=REWRITE_USER_QUERY_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "clarified_user_query_rewrite"},
    )


def build_rewrite_from_user_revision_prompt(
    initial_query: str,
    proposed_query: str,
    clarification_question: str,
    clarification_reason: str,
    clarification_answer: str,
    user_revision: str,
) -> PromptSpec:
    """Prompt para refinar a revisão escrita pelo usuário."""

    text = f"""Retorne somente um objeto JSON. Sem markdown. Sem explicações fora do JSON.

Crie um único tema acadêmico claro a partir da revisão escrita pelo usuário.

Regras:
- Priorize a revisão escrita pelo usuário.
- Preserve contexto útil da consulta inicial e do esclarecimento somente quando for compatível com a revisão.
- Não invente métodos, datasets, domínios, restrições ou modalidades.
- Não crie múltiplas queries.
- proposed_query deve permanecer em inglês porque será usada na busca acadêmica.
- message deve ser em português porque será mostrado ao usuário.
- reason deve ser em português para facilitar depuração.
- Mantenha as chaves JSON em inglês.
- proposed_query deve ser direta, fiel e não excessivamente longa.

Consulta inicial: {initial_query}
Tema proposto anteriormente: {proposed_query}
Pergunta de esclarecimento: {clarification_question}
Justificativa do esclarecimento: {clarification_reason}
Resposta ao esclarecimento: {clarification_answer}
Revisão do usuário: {user_revision}

Formato JSON obrigatório:
{{"proposed_query":"tema acadêmico claro em inglês","message":"Com base na sua revisão, a busca ficaria: ...","reason":"justificativa curta em português"}}
"""

    return PromptSpec(
        name="rewrite_from_user_revision",
        version=REWRITE_FROM_USER_REVISION_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "user_revision_query_rewrite"},
    )


def build_initial_queries_prompt(context: SearchContext) -> PromptSpec:
    """Prompt para planejar as queries iniciais."""

    text = f"""Retorne somente um objeto JSON. Sem markdown. Sem explicações fora do JSON.

Você está planejando a primeira rodada de queries para busca de artigos acadêmicos.

Tema do usuário:
{context.user_query}

Ano mínimo:
{context.min_year}

Tarefa:
Crie exatamente 3 queries acadêmicas para encontrar artigos candidatos.
O tema já foi avaliado ou refinado. Não faça perguntas e não reavalie o contexto.

Filosofia de busca:
Priorize recall, pois uma etapa posterior fará validação semântica.
A busca deve ser ampla o suficiente para descobrir artigos úteis, mas fiel ao tema.

Planejamento interno:
1. Identifique a intenção principal.
2. Identifique os núcleos obrigatórios que devem aparecer em todas as queries, exatamente ou por sinônimos acadêmicos próximos.
3. Identifique termos técnicos opcionais.
4. Identifique sinônimos acadêmicos prováveis.
5. Identifique termos que causariam desvio de tema.

Núcleos obrigatórios:
- Geralmente são a tarefa, objeto, modalidade, domínio ou problema sem os quais a query deixa de representar o tema.
- Exemplo: audio violence detection exige violence detection e audio/sound/acoustic.
- Exemplo: bias in medical image classification exige bias e medical image classification/medical imaging classification.
- Exemplo: shopping agents exige shopping/automated shopping e agents/assistants.

Termos opcionais:
Podem aparecer em algumas queries, não necessariamente em todas.
Exemplos: LLM, VLM, RAG, LSTM, transformer, benchmark, dataset, state of the art, trends, survey.

Quando o tema indicar "general", "geral", "visão geral", "overview", "broad" ou "panorama":
- mantenha a busca ampla;
- exatamente uma query pode incluir "state of the art", "trends", "overview", "survey" ou "review";
- não use esses termos em todas as queries, salvo pedido explícito.

Estrutura:
- Query 1: formulação acadêmica mais fiel.
- Query 2: sinônimo ou formulação acadêmica equivalente.
- Query 3: ângulo exploratório mais amplo, mas ainda fiel.

Regras:
- O array queries deve conter exatamente 3 strings não vazias.
- As queries acadêmicas devem obrigatoriamente permanecer em inglês.
- reason deve ser em português para facilitar depuração.
- Mantenha as chaves JSON em inglês.
- Toda query deve preservar a intenção principal.
- Não invente datasets, métricas, anos, restrições ou aplicações estreitas.
- Não adicione método ou modalidade que não estejam no tema.
- Não amplie um tema específico para um campo genérico não relacionado.
- Não restrinja um tema amplo a um único método.
- Não copie exemplos sem verificar se servem ao tema atual.

Exemplos:

Tema: general audio-based violence detection
Saída:
{{"queries":["audio-based violence detection","acoustic violent event detection","state of the art sound-based violence detection"],"reason":"As queries preservam os núcleos de áudio e detecção de violência, variando termos acadêmicos próximos e usando uma formulação mais ampla para exploração geral."}}

Tema: shopping agents using LLMs and VLMs
Saída:
{{"queries":["AI shopping agents","LLM agents for online shopping","VLM shopping assistants in e-commerce"],"reason":"As queries preservam shopping agents e distribuem LLM e VLM como termos técnicos importantes sem tornar todas as formulações idênticas."}}

Tema: bias in medical image classification
Saída:
{{"queries":["bias in medical image classification","dataset bias in medical imaging classification","shortcut learning in medical image classification"],"reason":"As queries preservam bias e medical image classification, explorando terminologia próxima sobre viés."}}

Formato JSON obrigatório:
{{"queries":["academic search query 1","academic search query 2","academic search query 3"],"reason":"justificativa curta em português"}}
"""

    return PromptSpec(
        name="initial_queries",
        version=INITIAL_QUERIES_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "initial_query_planning"},
    )


def build_plan_filters_prompt(context: SearchContext) -> PromptSpec:
    """Prompt para planejar filtros semânticos."""

    text = f"""Retorne somente um objeto JSON válido. Sem markdown. Sem explicações fora do JSON.

Você está planejando filtros semânticos para validação de artigos acadêmicos.

Tema refinado do usuário:
{context.user_query}

Tarefa:
Crie filtros semânticos compactos a partir do tema refinado.
Esses filtros serão usados para decidir se artigos candidatos correspondem à intenção acadêmica.

Importante:
- Não gere queries de busca.
- Não valide artigos nesta etapa.
- Não invente um novo tema.
- Não renomeie chaves JSON.
- Inclua todas as chaves obrigatórias e não adicione chaves extras.
- Todos os campos, exceto primary_intent e reason, devem ser listas.
- Se uma lista não tiver valores, retorne [].

Regra de idioma:
- Mantenha as chaves JSON em inglês.
- Valores técnicos podem preservar termos acadêmicos em inglês quando isso aumentar a precisão.
- reason deve ser em português para facilitar depuração.

Schema JSON oficial:
{{
  "primary_intent": "string",
  "conservative_filters": ["string"],
  "expansive_filters": ["string"],
  "negative_constraints": ["string"],
  "not_inferred": ["string"],
  "validation_priority": ["string"],
  "reason": "string"
}}

Significado:
- primary_intent: intenção acadêmica central.
- conservative_filters: critérios centrais para relevância direta.
- expansive_filters: conceitos próximos que ajudam, mas não bastam sozinhos.
- negative_constraints: conceitos, domínios, tarefas ou interpretações que indicam incompatibilidade.
- not_inferred: elementos que não podem ser presumidos.
- validation_priority: ordem dos critérios de validação.
- reason: justificativa curta em português.

Princípio central:
O artigo deve primeiro satisfazer os conservative_filters para relevância direta.
Expansive_filters são apenas evidência de apoio.
Não aceite artigo apenas por compartilhar uma palavra isolada.

Filtros conservadores:
- Use somente tarefa, objeto, sistema, modalidade, domínio ou método central explícito ou fortemente implícito.
- Mantenha-os estreitos e centrais.
- Não use termos genéricos isolados como AI, machine learning, automation ou data analytics.
- Combine termos amplos com o objeto real: "shopping agent", não apenas "agent"; "audio-based violence detection", não apenas "audio".

Filtros expansivos:
- Use conceitos próximos para melhorar recall.
- Não adicione negócios, marketing, logística ou conceitos sociais sem solicitação.
- Não invente detalhes de implementação.

Restrições negativas:
- Inclua confusões prováveis causadas por termos ambíguos.
- Inclua tópicos próximos que produziriam artigos irrelevantes.
- Inclua domínios que compartilham palavras, mas não correspondem à intenção.

not_inferred:
Use para suposições tentadoras, mas não fornecidas, como dataset, arquitetura, plataforma, estudo de usuário, web scraping, API, implantação em tempo real, aplicativo ou recommender system.

Relevância em Ciência da Computação:
Para temas computacionais, prefira contribuições como algoritmos, modelos, agentes, sistemas, datasets, benchmarks, experimentos, arquiteturas, pipelines, software ou avaliação técnica.
Não force machine learning ou deep learning sem apoio no tema.

Limites:
- conservative_filters: máximo 4 itens.
- expansive_filters: máximo 8 itens.
- negative_constraints: máximo 10 itens.
- not_inferred: máximo 8 itens.
- validation_priority: máximo 5 itens.
- reason: uma frase curta em português.

Exemplos:

Tema: general audio-based violence detection
Saída:
{{
  "primary_intent": "general search about audio-based violence detection",
  "conservative_filters": ["violence detection", "audio-based detection"],
  "expansive_filters": ["sound-based violent event detection", "acoustic event detection", "audio signal processing", "audio classification", "aggression detection"],
  "negative_constraints": ["video-only violence detection", "image-only violence detection", "visual surveillance", "audio-visual violence detection", "multimodal violence detection", "hate speech detection", "deepfake detection", "non-computational violence studies"],
  "not_inferred": ["specific dataset", "specific model architecture", "real-time deployment", "multimodal learning"],
  "validation_priority": ["match violence detection task", "match audio modality", "check computational contribution", "apply negative constraints", "use expansive filters only as supporting evidence"],
  "reason": "O tema exige detecção de violência baseada em áudio, sem presumir método ou dataset específico."
}}

Tema: shopping agents using LLMs and VLMs
Saída:
{{
  "primary_intent": "academic search about shopping agents using LLMs or VLMs",
  "conservative_filters": ["shopping agent", "LLM or VLM agent", "online shopping or purchasing assistance"],
  "expansive_filters": ["AI shopping assistant", "multimodal shopping assistant", "product search agent", "product comparison", "price comparison", "web automation for shopping", "browser agent"],
  "negative_constraints": ["generic LLM agents without shopping context", "generic VLM papers without shopping context", "warehouse automation", "inventory optimization", "supply chain management", "principal-agent economics"],
  "not_inferred": ["specific dataset", "specific benchmark", "specific e-commerce platform", "web scraping", "API integration", "checkout automation"],
  "validation_priority": ["match shopping agent intent", "match LLM or VLM agent aspect", "check shopping or purchasing context", "check computational contribution", "apply negative constraints"],
  "reason": "O tema exige contexto de shopping agent e uso de LLM ou VLM."
}}

Agora retorne o objeto JSON para o tema refinado atual.

Formato JSON obrigatório:
{{
  "primary_intent": "string",
  "conservative_filters": ["string"],
  "expansive_filters": ["string"],
  "negative_constraints": ["string"],
  "not_inferred": ["string"],
  "validation_priority": ["string"],
  "reason": "justificativa curta em português"
}}
"""

    return PromptSpec(
        name="plan_filters",
        version=PLAN_FILTERS_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "semantic_filter_planning"},
    )


def build_validate_papers_prompt(
    context: SearchContext,
    papers: list[Paper],
    search_feedback: dict[str, object] | None = None,
    search_filters: dict[str, object] | None = None,
) -> PromptSpec:
    """Prompt para validação semântica de artigos candidatos."""

    feedback = search_feedback or {}
    filters = search_filters or {}

    revised_topic = _feedback_text(feedback, "revised_topic") or context.user_query
    positive_constraints = _feedback_list(feedback, "positive_constraints")
    feedback_negative_constraints = _feedback_list(feedback, "negative_constraints")
    query_strategy = _feedback_text(feedback, "query_strategy") or "nenhuma"

    primary_intent = _feedback_text(filters, "primary_intent") or "nenhuma"
    conservative_filters = _feedback_list(filters, "conservative_filters")
    expansive_filters = _feedback_list(filters, "expansive_filters")
    negative_constraints = [
        *_feedback_list(filters, "negative_constraints"),
        *feedback_negative_constraints,
    ]
    not_inferred = _feedback_list(filters, "not_inferred")
    validation_priority = _feedback_list(filters, "validation_priority")

    # Fallback compatível com versões antigas do schema de filtros.
    if not conservative_filters:
        conservative_filters = [
            *_feedback_list(filters, "required_concepts"),
            *_feedback_list(filters, "required_modality"),
        ]
    if not expansive_filters:
        expansive_filters = [
            *_feedback_list(filters, "positive_signals"),
            *_feedback_list(filters, "soft_preferences"),
        ]
    if not negative_constraints:
        negative_constraints = [
            *_feedback_list(filters, "negative_signals"),
            *_feedback_list(filters, "hard_exclusion_rules"),
        ]

    paper_block = _format_candidate_papers(papers)
    candidate_count = len(papers)
    candidate_label = "artigo candidato" if candidate_count == 1 else "artigos candidatos"

    text = f"""Retorne somente um objeto JSON. Sem markdown. Sem explicações fora do JSON.

Você está validando semanticamente o lote atual de resultados acadêmicos.
O lote contém exatamente {candidate_count} {candidate_label}.

Intenção de busca do usuário:
{context.user_query}

Tema revisado atual:
{revised_topic}

Restrições positivas do feedback:
{_format_items(positive_constraints)}

Restrições negativas do feedback:
{_format_items(feedback_negative_constraints)}

Estratégia de busca:
{query_strategy}

Filtros semânticos planejados:
Intenção principal: {primary_intent}
Filtros conservadores: {_format_items(conservative_filters)}
Filtros expansivos: {_format_items(expansive_filters)}
Restrições negativas: {_format_items(negative_constraints)}
Não inferir: {_format_items(not_inferred)}
Prioridade de validação: {_format_items(validation_priority)}

Lote atual:
{paper_block}

Tarefa:
Valide somente os {candidate_count} artigos deste lote.
Para cada candidato, decida sua relevância para a intenção do usuário.

Regra crítica de saída:
- Retorne exatamente {candidate_count} objetos dentro de validated_papers.
- Retorne um objeto para cada candidato, sem omitir ou adicionar artigos.
- paper_id deve corresponder exatamente ao id fornecido.
- Não use título, DOI, URL, índice ou identificador inventado como paper_id.

Princípio central:
Seja baseado em evidência e calibrado.
Inclua um artigo somente quando título ou abstract sustentarem uma relação real com a intenção.
Use high para correspondência direta, medium para correspondência parcial forte, low para base útil e reject para tema incorreto.
Palavras genéricas isoladas não provam relevância.

Antes de validar cada artigo:
1. Identifique o tema central da busca.
2. Identifique tarefa, modalidade, domínio, método ou aplicação exigidos.
3. Compare título e abstract com esses requisitos.
4. Classifique como diretamente relevante, parcialmente relevante, base útil ou irrelevante.

Gates para relevância direta:
Antes de high ou medium, verifique:
1. correspondência com primary_intent;
2. atendimento aos conservative_filters;
3. ausência de violação de negative_constraints;
4. ausência de suposições de not_inferred;
5. contribuição computacional clara, quando o tema for computacional.

Se falhar em um gate:
- não atribua high;
- use medium somente se faltar um aspecto secundário;
- use low/include somente se houver base claramente útil por método, modalidade, tipo de dado ou família de tarefas relacionada;
- use reject/exclude para tema, domínio, modalidade ou tarefa incorretos, ou mera coincidência de palavras genéricas.

Filtros:
- Conservative filters definem relevância direta.
- Expansive filters são apoio e podem sustentar low/background quando houver relação genuína.
- Expansive filters não resgatam domínio, modalidade ou tarefa claramente incorretos.
- Negative constraints exigem exclusão quando claramente presentes.

Calibração de importância:
- Use a escala completa.
- high: artigo provavelmente central para o tema.
- medium: artigo claramente útil e relacionado, mas não perfeito.
- low: artigo útil como base, método, conceito, dataset, avaliação, modalidade ou família de tarefas próxima.
- reject: artigo de tema, domínio, modalidade ou tarefa incorretos, ou apenas com palavras genéricas.

Regra de evidência:
- relevance_reason deve ser específica para o candidato.
- Cite evidência concreta do título ou abstract por paráfrase curta.
- Não reutilize justificativas entre artigos diferentes.
- Não diga que o título menciona X se X ou equivalente próximo não estiver no título.
- Não invente informações.
- Se o abstract estiver ausente, julgue com cautela pelo título e metadados.
- Todo artigo incluído deve ter relevance_reason compreensível e honesta.

Regra de idioma:
- relevance_reason, mismatch_reason, useful_for, summary e textos de justificativa devem ser em português para facilitar depuração.
- Mantenha as chaves JSON e os rótulos high, medium, low, reject, include e exclude em inglês.

Exemplos:
Os exemplos ensinam o padrão de raciocínio. Não copie o texto deles; adapte ao título e abstract do candidato atual.

Exemplo direto:
Intenção: explainable AI methods for machine learning models
Título: SHAP-Based Explanations for Tree-Based Machine Learning Models
Saída:
{{"relevance":"high","decision":"include","relevance_reason":"O título aborda explicações SHAP para modelos de machine learning baseados em árvores, correspondendo diretamente a métodos de explicabilidade.","mismatch_reason":"","useful_for":"métodos"}}

Exemplo de base:
Intenção: bias in medical image classification
Título: Generalization in Natural Image Classification Benchmarks
Saída:
{{"relevance":"low","decision":"include","relevance_reason":"O artigo não trata de imagens médicas, mas pode oferecer base sobre generalização em classificação de imagens.","mismatch_reason":"Não contém o contexto de medical imaging exigido para relevância direta.","useful_for":"base conceitual"}}

Exemplo de rejeição:
Intenção: shopping agents using LLMs and VLMs
Título: Deep Generative Modelling: A Comparative Review
Saída:
{{"relevance":"reject","decision":"exclude","relevance_reason":"","mismatch_reason":"O artigo revisa modelos generativos de forma geral e não apresenta evidência de shopping agents ou assistência de compra.","useful_for":"não útil"}}

Formato JSON obrigatório:
{{"validated_papers":[{{"paper_id":"paper id","relevance":"high","decision":"include","relevance_reason":"justificativa curta baseada em evidência","mismatch_reason":"","useful_for":"como ajuda a pesquisa"}}],"summary":"resumo curto dos tipos de artigos incluídos e rejeitados"}}
"""

    return PromptSpec(
        name="validate_papers",
        version=VALIDATE_PAPERS_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "paper_semantic_validation"},
    )


def build_judge_paper_validations_prompt(
    context: SearchContext,
    papers: list[Paper],
    validated_papers: list[dict[str, object]],
    search_filters: dict[str, object],
    search_feedback: dict[str, object] | None = None,
) -> PromptSpec:
    """Prompt para auditar validações semânticas de artigos."""

    feedback = search_feedback or {}
    filters = search_filters or {}
    revised_topic = _feedback_text(feedback, "revised_topic") or context.user_query
    conservative_filters = _feedback_list(filters, "conservative_filters")
    expansive_filters = _feedback_list(filters, "expansive_filters")
    negative_constraints = [
        *_feedback_list(filters, "negative_constraints"),
        *_feedback_list(feedback, "negative_constraints"),
    ]
    primary_intent = _feedback_text(filters, "primary_intent") or revised_topic
    candidate_count = len(papers)

    text = f"""Retorne somente um objeto JSON. Sem markdown. Sem explicações fora do JSON.

Você é um AI-as-a-judge independente auditando validações de artigos produzidas por outro modelo.
A validação original não é confiável por padrão.

Intenção de busca:
{context.user_query}

Tema revisado atual:
{revised_topic}

Filtros:
Intenção principal: {primary_intent}
Filtros conservadores: {_format_items(conservative_filters)}
Filtros expansivos: {_format_items(expansive_filters)}
Restrições negativas: {_format_items(negative_constraints)}

Artigos candidatos; use somente estas evidências:
{_format_candidate_papers_for_judge(papers)}

Validações originais:
{_format_validated_papers_for_judge(validated_papers)}

Tarefa:
Audite exatamente {candidate_count} artigos, um objeto por paper_id.
Verifique se decisão e justificativa originais são sustentadas pelo título ou abstract.
Corrija relevância e decisão quando necessário.

Antes de julgar cada artigo, identifique internamente:
1. assunto ou domínio central;
2. tarefa ou problema central;
3. modalidade, tipo de dado, método ou aplicação exigidos;
4. contribuição esperada, como método, modelo, algoritmo, sistema, survey, benchmark, dataset, experimento ou avaliação.
Não retorne esses requisitos separadamente.

Princípio:
Julgue o artigo, não a confiança do modelo anterior.
A validação original pode estar errada, exagerada, copiada ou sem sustentação.

Calibração de importância:
- Use a escala completa.
- high: artigo provavelmente central para o tema.
- medium: artigo claramente útil e relacionado, mas não perfeito.
- low: artigo útil como base, método, conceito, dataset, avaliação, modalidade ou família de tarefas próxima.
- reject: artigo de tema, domínio, modalidade ou tarefa incorretos, ou apenas com palavras genéricas.

Rejeição:
Use reject/exclude quando:
- modalidade ou domínio estiverem errados;
- a tarefa for claramente não relacionada;
- houver violação de restrição negativa;
- houver somente palavras genéricas;
- não houver utilidade metodológica, de modalidade ou de família de tarefas.

Filtros:
- Conservative filters são evidência forte para relevância direta.
- passes_conservative_filters pode ser false para artigos low/include usados como base.
- Expansive filters apenas apoiam a decisão.
- Expansive filters não resgatam domínio ou modalidade claramente incorretos.
- Negative constraints são mais fortes e normalmente exigem exclusão.

Auditoria da justificativa:
- Confira se relevance_reason é sustentada pelo título ou abstract.
- Se disser "o título menciona X", confirme que X ou equivalente realmente aparece.
- Se inventar evidência, use reason_is_supported=false.
- Se repetir frase genérica para artigos diferentes, audite cada artigo independentemente.
- Se exagerar a relevância, não rejeite automaticamente; reduza a relevância e escreva judge_reason mais precisa.
- Não invente evidência.
- Se não houver abstract, use o título com cautela.

Evidência inventada:
Se a justificativa original afirma que o título menciona um tema, mas o título não menciona esse tema nem equivalente próximo:
- validation_is_correct=false;
- reason_is_supported=false;
- corrected_relevance deve ser menor que a relevância original;
- corrected_decision deve refletir a evidência real.

Raciocínio específico por tema:
Não use regras fixas de domínio.
Para cada busca, infira os requisitos centrais a partir da query atual, tema revisado, primary_intent, conservative_filters, expansive_filters e negative_constraints.

Calibração de importância:
- Use a escala completa.

Correção do juiz:
- Se a validação original diz high/include, mas o artigo é apenas parcialmente relacionado, use medium ou low; não transforme tudo em reject.
- Se a validação original diz reject/exclude, mas o artigo é útil como base, use low/include.
- Use reject somente quando não houver relação útil ou houver violação clara.

Exemplos ilustrativos:
Não trate os exemplos como regras fixas. Use-os apenas como padrão de raciocínio.

Exemplo A:
Tema: audio-based violence detection
Título: Supervised machine learning for audio emotion recognition
Saída esperada:
- validation_is_correct=false
- reason_is_supported=false
- passes_conservative_filters=false
- violates_negative_constraints=false
- corrected_relevance="low"
- corrected_decision="include"
- judge_reason: "O artigo não trata de detecção de violência, mas estuda machine learning supervisionado para reconhecimento de emoção em áudio, podendo fornecer base metodológica para reconhecimento baseado em áudio."

Exemplo B:
Tema: audio-based violence detection
Título: Violence Detection in Videos by Combining 3D CNNs and SVM
Saída esperada:
- validation_is_correct=false
- reason_is_supported=false
- passes_conservative_filters=false
- violates_negative_constraints=true
- corrected_relevance="reject"
- corrected_decision="exclude"
- judge_reason: "O artigo trata de detecção de violência em vídeo, não de detecção de violência baseada em áudio."

Exemplo C:
Tema: modern audio classification techniques
Título: Audio classification using self-supervised learning representations
Saída esperada:
- validation_is_correct=true
- reason_is_supported=true
- passes_conservative_filters=true
- violates_negative_constraints=false
- corrected_relevance="high"
- corrected_decision="include"
- judge_reason: "O artigo aborda diretamente classificação de áudio com uma técnica computacional moderna."

Exemplo D:
Tema: modern audio classification techniques
Título: A comprehensive survey on support vector machine classification
Saída esperada:
- validation_is_correct=false
- reason_is_supported=false
- passes_conservative_filters=false
- violates_negative_constraints=false
- corrected_relevance="reject"
- corrected_decision="exclude"
- judge_reason: "O artigo trata de métodos de classificação de forma geral, mas não estabelece audio classification como tema central."

Exemplo E:
Tema: explainable AI methods for machine learning models
Título: Fairness and Accountability in Automated Decision Systems
Saída esperada:
- corrected_relevance="low"
- corrected_decision="include"
- judge_reason: "O artigo não trata diretamente de métodos de explicabilidade, mas pode fornecer base de contexto sobre responsible AI se a busca permitir uma visão mais ampla."

Regras obrigatórias:
- Avalie alinhamento semântico, não palavras isoladas.
- Não preserve a decisão original por cortesia.
- validation_is_correct é true somente se relevância e decisão originais estiverem corretas.
- passes_conservative_filters é true somente para atendimento direto aos critérios centrais.
- passes_conservative_filters pode ser false para low/include.
- violates_negative_constraints é true quando houver restrição negativa ou interpretação de domínio incorreto.
- Se violates_negative_constraints=true, corrected_relevance deve ser "reject" e corrected_decision deve ser "exclude".
- judge_reason deve ser uma justificativa curta, concreta e baseada em evidência, em português.
- summary deve ser em português para facilitar depuração.
- Mantenha chaves JSON e rótulos em inglês.

Rótulos permitidos:
- corrected_relevance: "high", "medium", "low" ou "reject"
- corrected_decision: "include" ou "exclude"

Regras críticas de saída:
- Retorne exatamente {candidate_count} objetos em judged_validations.
- Retorne um objeto para cada candidato e nenhum artigo adicional.
- paper_id deve corresponder exatamente ao id fornecido.
- Campos booleanos devem ser booleanos JSON.
- Não adicione chaves extras.

Formato JSON obrigatório:
{{"judged_validations":[{{"paper_id":"paper id","validation_is_correct":true,"reason_is_supported":true,"passes_conservative_filters":true,"violates_negative_constraints":false,"corrected_relevance":"high","corrected_decision":"include","judge_reason":"justificativa curta em português"}}],"summary":"resumo curto em português"}}
"""

    return PromptSpec(
        name="judge_paper_validations",
        version=JUDGE_PAPER_VALIDATIONS_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "paper_validation_audit"},
    )


def build_refine_queries_prompt(
    context: SearchContext,
    validated_papers: list[dict[str, object]],
    used_queries: list[str],
    search_feedback: dict[str, object] | None = None,
) -> PromptSpec:
    """Prompt para refinar queries após validação e feedback."""

    evidence = _format_validated_papers(validated_papers[:12])
    used = "; ".join(used_queries[-8:]) or "nenhuma"
    feedback = search_feedback or {}
    revised_topic = _feedback_text(feedback, "revised_topic") or context.user_query
    positive_constraints = _feedback_list(feedback, "positive_constraints")
    negative_constraints = _feedback_list(feedback, "negative_constraints")
    query_strategy = _feedback_text(feedback, "query_strategy") or "nenhuma"

    text = f"""Retorne somente um objeto JSON. Sem markdown. Sem explicações fora do JSON.

Você está planejando uma nova rodada de queries acadêmicas após a validação semântica dos resultados anteriores.

Tema original:
{context.user_query}

Tema revisado atual:
{revised_topic}

Ano mínimo:
{context.min_year}

Queries já usadas:
{used}

Evidências dos artigos validados:
{evidence}

Restrições positivas:
{_format_items(positive_constraints)}

Restrições negativas:
{_format_items(negative_constraints)}

Estratégia de busca:
{query_strategy}

Tarefa:
Crie exatamente 3 novas queries acadêmicas.
Elas devem melhorar recall e evitar os erros encontrados nos artigos rejeitados.
Não repita queries já usadas.

Estratégia:
- Use artigos high/medium incluídos como evidência de terminologia útil.
- Use artigos low/include com cautela, apenas como inspiração secundária.
- Use artigos rejeitados e mismatch_reason para identificar o que evitar.
- Corrija desvios de domínio, modalidade ou tarefa.
- Se os resultados anteriores foram estreitos demais, explore sinônimos próximos.
- Se foram amplos demais, use termos acadêmicos mais precisos.
- Priorize o feedback do usuário sobre evidências anteriores.

Núcleos e termos opcionais:
- Identifique os núcleos obrigatórios do tema revisado e do feedback.
- Esses núcleos devem aparecer em todas as queries, exatamente ou por sinônimos próximos.
- Termos técnicos opcionais podem aparecer em apenas algumas queries.
- As três queries devem ser significativamente diferentes.

Tema geral:
Se o tema revisado for geral, uma query pode usar "state of the art", "trends", "overview", "survey" ou "review".
Não use esses termos em todas, salvo pedido explícito por reviews/surveys.

Regras:
- queries deve conter exatamente 3 strings não vazias.
- As queries acadêmicas devem obrigatoriamente permanecer em inglês.
- reason deve ser em português para facilitar depuração.
- Mantenha as chaves JSON em inglês.
- Toda query deve seguir o tema revisado, restrições e estratégia.
- Evite termos das negative_constraints.
- Não invente datasets, métricas, domínios, modalidades, métodos, aplicações ou intenção de review.
- Não use títulos rejeitados como inspiração positiva.

Exemplo:
Tema revisado: general audio-based violence detection
Evidência rejeitada: video-based violence detection; hate speech detection; deepfake detection
Saída:
{{"queries":["audio-based violence detection","acoustic violent event detection","state of the art sound-based violence detection"],"reason":"As novas queries preservam os núcleos de áudio e detecção de violência, evitando desvios para vídeo, texto e deepfakes."}}

Formato JSON obrigatório:
{{"queries":["academic search query 1","academic search query 2","academic search query 3"],"reason":"justificativa curta em português"}}
"""

    return PromptSpec(
        name="refine_queries",
        version=REFINE_QUERIES_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "query_refinement"},
    )


def build_feedback_analysis_prompt(
    original_query: str,
    refined_query: str,
    user_feedback: str,
    validated_papers: list[dict[str, object]],
    used_queries: list[str],
) -> PromptSpec:
    """Prompt para interpretar feedback humano nos artigos validados."""

    papers = _format_validated_papers(validated_papers[:20])
    used = "; ".join(used_queries[-10:]) or "nenhuma"

    text = f"""Retorne somente um objeto JSON. Sem markdown. Sem explicações fora do JSON.

Você está melhorando uma busca acadêmica após o usuário revisar os artigos validados.

Tarefa:
Interprete a crítica do usuário como orientação estruturada antes de gerar novas queries.

Entradas:
Consulta original: {original_query}
Consulta refinada atual: {refined_query}
Feedback do usuário: {user_feedback}
Queries usadas: {used}
Evidências dos artigos validados:
{papers}

Regras:
- Não gere queries nesta etapa.
- Não copie o feedback bruto diretamente para query_strategy.
- Converta o feedback em um revised_topic acadêmico mais limpo.
- revised_topic deve permanecer em inglês porque será usado para gerar queries.
- positive_constraints, negative_constraints, query_strategy e reason devem ser em português para facilitar depuração.
- Mantenha as chaves JSON em inglês.
- Preserve a intenção original, salvo mudança explícita do usuário.
- Coloque conceitos, modalidades, métodos, domínios ou termos desejados em positive_constraints.
- Coloque conceitos, modalidades, métodos, domínios ou interpretações excluídos em negative_constraints.
- query_strategy deve ser uma frase acionável para a próxima etapa.
- Mantenha cada restrição curta e pesquisável.
- Use artigos incluídos e excluídos como evidência do que funcionou e falhou.
- Se o usuário esclarecer termo ambíguo, selecione o significado correto e exclua os incorretos.

Exemplo:
Consulta original: audio violence detection
Consulta refinada: general audio-based violence detection
Feedback: na verdade era para ser apenas audio, sem multimodal e sem audio-visual
Saída:
{{"revised_topic":"general audio-only violence detection","positive_constraints":["detecção de violência somente por áudio","detecção baseada em som","sinais acústicos"],"negative_constraints":["multimodal","audio-visual","audiovisual","detecção baseada em vídeo","vigilância visual"],"query_strategy":"Buscar artigos em que áudio, som ou sinais acústicos sejam a entrada central e evitar trabalhos multimodais, audiovisuais ou baseados em vídeo.","reason":"O usuário esclareceu que a modalidade pretendida é somente áudio e rejeitou resultados multimodais ou audiovisuais."}}

Formato JSON obrigatório:
{{"revised_topic":"clean revised academic topic in English","positive_constraints":["restrição em português"],"negative_constraints":["restrição em português"],"query_strategy":"estratégia acionável em português","reason":"justificativa curta em português"}}
"""

    return PromptSpec(
        name="feedback_analysis",
        version=FEEDBACK_ANALYSIS_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "search_feedback_analysis"},
    )


def build_continue_decision_prompt(
    context: SearchContext,
    round_number: int,
    validated_papers: list[dict[str, object]],
    last_new_paper_count: int,
    last_new_useful_count: int,
) -> PromptSpec:
    """Prompt para decidir se outra rodada é útil."""

    evidence = _format_validated_papers(validated_papers[:10])

    text = f"""Retorne somente um objeto JSON. Sem markdown. Sem explicações fora do JSON.

Decida se outra rodada de busca acadêmica é útil.

Tema: {context.user_query}
Rodada: {round_number} de {context.max_rounds}
Novos artigos: {last_new_paper_count}
Novos artigos úteis validados: {last_new_useful_count}
Evidências dos artigos validados:
{evidence}

Regras:
- Continue se houver poucos artigos high/medium úteis e as rejeições indicarem um desvio corrigível nas queries.
- Continue se os artigos low indicarem uma direção promissora que ainda precisa de queries mais precisas.
- Pare se já houver artigos úteis suficientes.
- Pare se a última rodada repetir os mesmos erros e não houver direção nova clara.
- reason deve ser em português para facilitar depuração.
- Mantenha as chaves JSON em inglês.

Formato JSON obrigatório:
{{"continue":true,"reason":"justificativa curta em português"}}
"""

    return PromptSpec(
        name="continue_decision",
        version=CONTINUE_DECISION_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "continue_decision"},
    )


def _feedback_text(feedback: dict[str, object], key: str) -> str:
    value = feedback.get(key)
    if value is None:
        return ""
    return " ".join(str(value).split())


def _feedback_list(feedback: dict[str, object], key: str) -> list[str]:
    value = feedback.get(key)
    if not isinstance(value, list):
        return []

    items: list[str] = []
    for item in value:
        text = " ".join(str(item).split())
        if text:
            items.append(text)
    return items


def _format_items(items: list[str]) -> str:
    return "; ".join(items) if items else "nenhum"


def _format_candidate_papers(papers: list[Paper]) -> str:
    if not papers:
        return "nenhum"

    lines: list[str] = []
    for position, paper in enumerate(papers, start=1):
        abstract = " ".join((paper.abstract or "").split())[:700] or "sem abstract"
        lines.append(
            f"{position}. id={paper.id}; título={paper.title}; "
            f"ano={paper.year or 'desconhecido'}; fonte={paper.source}; "
            f"url={paper.url or 'nenhuma'}; abstract={abstract}"
        )
    return "\n".join(lines)


def _format_validated_papers(validated_papers: list[dict[str, object]]) -> str:
    if not validated_papers:
        return "nenhum"

    lines: list[str] = []
    for position, item in enumerate(validated_papers, start=1):
        paper = item.get("paper")
        title = getattr(paper, "title", "desconhecido")
        paper_id = getattr(paper, "id", item.get("paper_id", "desconhecido"))
        relevance = _feedback_text(item, "relevance") or "desconhecida"
        decision = _feedback_text(item, "decision") or "desconhecida"
        relevance_reason = _feedback_text(item, "relevance_reason")
        mismatch_reason = _feedback_text(item, "mismatch_reason")
        lines.append(
            f"{position}. id={paper_id}; título={title}; relevance={relevance}; "
            f"decision={decision}; motivo_inclusão={relevance_reason or 'nenhum'}; "
            f"motivo_exclusão={mismatch_reason or 'nenhum'}"
        )
    return "\n".join(lines)


def _format_validated_papers_for_judge(
    validated_papers: list[dict[str, object]],
) -> str:
    if not validated_papers:
        return "nenhum"

    lines: list[str] = []
    for position, item in enumerate(validated_papers, start=1):
        paper = item.get("paper")
        paper_id = getattr(paper, "id", item.get("paper_id", "desconhecido"))
        lines.append(
            f"{position}. paper_id={paper_id}; "
            f"relevance={_feedback_text(item, 'relevance') or 'desconhecida'}; "
            f"decision={_feedback_text(item, 'decision') or 'desconhecida'}; "
            f"relevance_reason={_feedback_text(item, 'relevance_reason') or 'nenhuma'}; "
            f"mismatch_reason={_feedback_text(item, 'mismatch_reason') or 'nenhuma'}; "
            f"useful_for={_feedback_text(item, 'useful_for') or 'nenhum'}"
        )
    return "\n".join(lines)


def _format_candidate_papers_for_judge(papers: list[Paper]) -> str:
    if not papers:
        return "nenhum"

    lines: list[str] = []
    for position, paper in enumerate(papers, start=1):
        abstract = " ".join((paper.abstract or "").split())[:700] or "sem abstract"
        lines.append(
            f"{position}. paper_id={paper.id}; título={paper.title}; abstract={abstract}"
        )
    return "\n".join(lines)
