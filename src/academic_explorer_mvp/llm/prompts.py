from dataclasses import dataclass, field

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.paper import Paper


@dataclass(frozen=True)
class PromptSpec:
    """Versioned prompt used by the local model."""

    name: str
    version: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"


INITIAL_QUERIES_PROMPT_VERSION = "1.0.5"
PLAN_FILTERS_PROMPT_VERSION = "1.0.1"
REFINE_QUERIES_PROMPT_VERSION = "1.0.3"
FEEDBACK_ANALYSIS_PROMPT_VERSION = "1.0.0"
VALIDATE_PAPERS_PROMPT_VERSION = "1.0.1"
CONTINUE_DECISION_PROMPT_VERSION = "1.0.0"
ENRICH_QUERY_PROMPT_VERSION = "1.0.1"
ASSESS_QUERY_CONTEXT_PROMPT_VERSION = "1.0.0"
CONTEXT_QUESTION_PROMPT_VERSION = "1.0.0"
REWRITE_USER_QUERY_PROMPT_VERSION = "1.0.0"
REWRITE_FROM_USER_REVISION_PROMPT_VERSION = "1.0.0"


def build_assess_query_context_prompt(context: SearchContext) -> PromptSpec:
    """Prompt for deciding whether the query has enough context."""

    text = f"""Return only one JSON object. No markdown. No explanation.

You are preparing an academic paper search.

Task:
Decide whether the user's query has enough context to start building useful academic search queries.

Language rule:
The reason must be in English because it is internal.
Keep JSON keys in English.

Core decision criterion:
A query has enough context only when it contains:

1. a recognizable academic topic; and
2. enough search intent to avoid guessing what the user wants.

Searchable is not the same as enough context.
If a general search can be created, but it would require assuming the user's desired focus, then context is missing.

Use has_enough_context=true when:

* the user explicitly asks for a review, survey, overview, introduction, state of the art, or broad learning;
* or the query contains a clear topic plus a specific research focus, such as method, dataset, data source, metric, comparison, restriction, application, domain, modality, or specific problem;
* and the next step can generate academic search queries without guessing the user's intended direction.

Use has_enough_context=false when:

* the query is broad and the user did not say whether they want a general overview or a specific focus;
* the query could naturally lead to several different academic searches;
* the query is searchable, but generating queries would require assuming the user's desired method, domain, data source, modality, application, metric, or review intent;
* the topic has multiple common academic meanings and the intended meaning is unclear.

Important:
Do not mark a query as sufficient only because general search queries can be generated.
Ask for clarification when the user's desired search direction is unknown.

Examples:

User query: stock prediction
Output:
{{"has_enough_context":false,"reason":"the query is searchable, but it does not specify whether the user wants a review or a specific focus such as methods, data sources, assets, markets, time horizon, or metrics"}}

User query: stock prediction review
Output:
{{"has_enough_context":true,"reason":"the user explicitly indicated review intent for the topic"}}

User query: stock price prediction using LSTM and news sentiment
Output:
{{"has_enough_context":true,"reason":"the query specifies the task, method, and data source"}}

User query: fraud detection
Output:
{{"has_enough_context":false,"reason":"the query is broad and does not specify whether the user wants a general overview or a specific focus such as data type, datasets, models, domain, or deployment"}}

User query: fraud detection review
Output:
{{"has_enough_context":true,"reason":"the user explicitly indicated review intent for the topic"}}

User query: image-based plant disease detection
Output:
{{"has_enough_context":false,"reason":"the query specifies the modality, but not whether the user wants a general overview or a specific focus such as datasets, models, features, metrics, real-time detection, or comparison"}}

User query: image-based plant disease detection review
Output:
{{"has_enough_context":true,"reason":"the user explicitly indicated review intent for image-based plant disease detection"}}

User query: image-based plant disease detection using deep learning
Output:
{{"has_enough_context":true,"reason":"the query specifies the task, modality, and method family"}}

User query: prompt engineering
Output:
{{"has_enough_context":false,"reason":"the query is broad and does not specify whether the user wants an overview or a specific focus such as LLMs, image generation, code generation, evaluation, or optimization"}}

User query: prompt engineering overview
Output:
{{"has_enough_context":true,"reason":"the user explicitly indicated overview intent for the topic"}}

User query: prompt engineering for image generation
Output:
{{"has_enough_context":false,"reason":"the query specifies the application domain, but not whether the user wants an overview or a specific focus such as style control, prompt evaluation, visual quality, or optimization"}}

User query: prompt engineering for image generation evaluation
Output:
{{"has_enough_context":true,"reason":"the query specifies the topic, application domain, and evaluation focus"}}

User query: noisy data in medical image classification
Output:
{{"has_enough_context":false,"reason":"the query is relevant, but noisy data may refer to label noise, image noise, acquisition noise, outliers, or robustness"}}

User query: label noise in medical image classification
Output:
{{"has_enough_context":true,"reason":"the query specifies the type of noise and the application domain"}}

User query: shortcut bias
Output:
{{"has_enough_context":false,"reason":"the query is a recognizable research problem, but does not specify whether the user wants an overview or a focus such as computer vision, NLP, medical imaging, dataset bias, or robustness"}}

User query: shortcut bias review
Output:
{{"has_enough_context":true,"reason":"the user explicitly indicated review intent for the topic"}}

User query: shortcut bias in computer vision
Output:
{{"has_enough_context":true,"reason":"the query specifies the research problem and application domain"}}


User query:
{context.user_query}

Required JSON shape when context is missing:
{{"has_enough_context":false,"reason":"short reason"}}

Required JSON shape when context is enough:
{{"has_enough_context":true,"reason":"short reason"}}
"""

    return PromptSpec(
        name="assess_query_context",
        version=ASSESS_QUERY_CONTEXT_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "query_context_assessment",
        },
    )


def build_context_question_prompt(
    user_query: str,
    reason: str,
    ) -> PromptSpec:
    """Prompt for generating one clarification question with domain-specific options."""

    text = f"""Return only one JSON object. No markdown. No explanation.

    You are preparing an academic paper search.

    Task:
    Generate exactly one useful clarification question for the user's current query.

    Important:
    Do not reassess whether the query has enough context.
    The assessment was already done.
    Your only task is to ask one question that helps complete the missing context.

    Current user query:
    {user_query}

    Reason from previous assessment:
    {reason}

    Core goal:
    Ask one question that helps identify the user's intended academic search direction.

    The initial query can be about any research topic.
    It may come from computer science, medicine, finance, education, biology, engineering, social sciences, design, or another academic area.

    Core rule:
    Do not copy examples literally.
    Examples are only references for the reasoning pattern.
    The user's real query may be very different from the examples.

    Your question must be built from:

    1. the actual user query;
    2. the missing context described in the reason;
    3. the academic area suggested by the query;
    4. natural search dimensions for that area.

    Reason handling rule:
    The reason is a hint, not a script.
    Do not copy the wording from the reason if it sounds unnatural.
    If the reason is too generic, infer the most useful missing dimension from the user query.
    If the reason suggests options unrelated to the query, ignore those options.

    How to build the question:

    1. Identify the real topic of the query.
    2. Identify the academic/research area of that topic.
    3. Identify what kind of information is missing.
    4. Generate options that are natural for that specific area.
    5. Prefer options that could become useful academic search terms.
    6. Ask one question only.

    Useful missing-context dimensions:
    Depending on the topic, the missing context may be:

    * review or overview intent;
    * method;
    * model family;
    * application;
    * domain;
    * modality;
    * dataset;
    * data source;
    * metric;
    * comparison;
    * population;
    * disease;
    * task;
    * system type;
    * time horizon;
    * deployment context;
    * evaluation focus;
    * theoretical vs practical focus.

    Do not force all dimensions into the question.
    Choose only the dimensions that make sense for the current topic.

    Generalization rule:
    If the topic does not match any known example or category, do not guess randomly.
    Instead:

    * extract the main nouns and technical terms from the query;
    * identify the academic field they belong to;
    * ask whether the user wants a general overview or a focus based on natural dimensions of that field.

    Question style:
    The question should be direct, useful, and specific.

    Good question patterns:

    * "Você quer uma visão geral sobre [topic] ou quer focar em algo mais específico, como [option 1], [option 2], [option 3] ou [option 4]?"
    * "Para [topic], você quer priorizar [option 1], [option 2], [option 3] ou uma revisão geral da área?"
    * "Em [topic], o foco deve ser [option 1], [option 2], [option 3], [option 4] ou uma visão geral?"

    Bad questions. Never ask:

    * What specific aspect do you need?
    * Can you provide more context?
    * What do you want to know?
    * Please clarify your query.
    * Could you be more specific?

    Avoid unrelated options:

    * Do not ask about audio, video, image, or text for stock prediction unless the query explicitly mentions multimodal data.
    * Do not ask about stock assets for medical image classification.
    * Do not ask about organs or exams for fake news detection.
    * Do not ask about modality if the modality is already present in the query.
    * Do not add methods, datasets, metrics, or domains that are not natural for the topic.

    Language rule:
    The question must be in Portuguese because it is shown to the user.
    The reason must be in English because it is internal.
    Keep JSON keys in English.

    Topic guidance:
    These are not fixed templates.
    Use them only when they fit the current query.

    * For stock prediction, finance, market forecasting, or asset prediction:
    natural options include review/overview, forecasting methods, asset type, market, time horizon, data source, technical indicators, news sentiment, fundamentals, macroeconomic data, or evaluation metrics.

    * For detection or classification tasks:
    natural options include review/overview, modality, data type, datasets, models, feature extraction, real-time detection, deployment context, benchmark comparison, or evaluation metrics.

    * For medical image classification:
    natural options include review/overview, image type, disease, organ, dataset, model, label noise, robustness, segmentation/classification, clinical validation, or evaluation metrics.

    * For noisy data:
    natural options include review/overview, label noise, input noise, acquisition noise, outliers, robust training, uncertainty, data cleaning, or evaluation under noise.

    * For shortcut bias:
    natural options include review/overview, computer vision, NLP, medical imaging, dataset bias, spurious correlations, robustness, evaluation, or mitigation methods.

    * For recommendation systems:
    natural options include review/overview, collaborative filtering, content-based filtering, deep learning, cold start, evaluation metrics, fairness, explainability, or domain-specific recommendation.

    * For fake news detection:
    natural options include review/overview, text-based detection, social network propagation, multimodal detection, datasets, explainability, language, misinformation, or fact-checking.

    Examples:
    The examples below are illustrative.
    Do not copy their options unless they fit the current user query.
    Use them to understand how to connect the missing context with topic-specific academic search directions.

    User query: stock prediction
    Reason from previous assessment: the query is broad and does not specify whether the user wants a general review or a specific focus
    Output:
    {{"question":"Você quer uma visão geral sobre previsão de ações ou quer focar em algo mais específico, como métodos de previsão, tipo de ativo, horizonte temporal, fontes de dados ou métricas de avaliação?","reason":"the question offers finance-specific directions that would change the academic search terms"}}

    User query: stock prediction
    Reason from previous assessment: the query does not indicate which data source should guide the search
    Output:
    {{"question":"Na previsão de ações, você quer considerar quais fontes de dados: séries históricas de preços, indicadores técnicos, notícias, sentimento de mercado, fundamentos financeiros ou dados macroeconômicos?","reason":"the question asks about data sources that are relevant to stock prediction"}}

    User query: fraud detection
    Reason from previous assessment: the query is broad and does not specify whether the focus is a general review or a specific data type
    Output:
    {{"question":"Você quer uma visão geral sobre detecção de fraude ou quer focar em algo mais específico, como transações financeiras, grafos, dados tabulares, modelos, datasets ou métricas de avaliação?","reason":"the data type and research focus strongly change the search terms for fraud detection"}}

    User query: image-based plant disease detection
    Reason from previous assessment: the query has a modality, but does not specify whether the user wants a general review or a specific research focus
    Output:
    {{"question":"Você quer uma visão geral sobre detecção de doenças em plantas por imagem ou quer focar em algo mais específico, como datasets, modelos, extração de características, métricas de avaliação, implantação em campo ou comparação entre abordagens?","reason":"image already defines the modality, so the question focuses on the missing research direction"}}

    User query: noisy data in medical image classification
    Reason from previous assessment: noise may refer to labels, images, acquisition, or robustness
    Output:
    {{"question":"Em dados ruidosos para classificação de imagens médicas, você quer focar em rótulos ruidosos, ruído na imagem, ruído de aquisição, robustez do modelo ou uma revisão geral do tema?","reason":"different meanings of noise lead to different academic search terms"}}

    User query: prompt engineering
    Reason from previous assessment: the query is broad and does not specify application domain or research focus
    Output:
    {{"question":"Você quer uma visão geral sobre engenharia de prompt ou quer focar em algo mais específico, como modelos de linguagem, geração de imagens, geração de código, avaliação de prompts ou técnicas de otimização?","reason":"prompt engineering can involve different applications and research focuses"}}

    User query: prompt engineering for image generation
    Reason from previous assessment: the query specifies the application domain, but not the research focus
    Output:
    {{"question":"Em engenharia de prompt para geração de imagens, você quer uma visão geral da área ou quer focar em controle de estilo, qualidade visual, avaliação de prompts, modelos texto-imagem ou técnicas de otimização?","reason":"the topic already specifies image generation, so the question focuses on the missing research direction"}}

    User query: LLM hallucination
    Reason from previous assessment: the query is broad and does not specify whether the user wants causes, evaluation, or mitigation
    Output:
    {{"question":"Você quer uma visão geral sobre alucinação em LLMs ou quer focar em algo mais específico, como causas, métricas de avaliação, mitigação, RAG, verificação factual ou benchmarks?","reason":"LLM hallucination can be studied through causes, evaluation, mitigation, and system design"}}

    User query: machine learning for agriculture
    Reason from previous assessment: the query is broad and does not specify application or data type
    Output:
    {{"question":"Em machine learning para agricultura, você quer uma visão geral da área ou quer focar em aplicações específicas, como previsão de produtividade, detecção de doenças em plantas, irrigação inteligente, imagens de satélite ou sensores IoT?","reason":"machine learning in agriculture can involve different applications and data sources"}}

    User query: sentiment analysis
    Reason from previous assessment: the query is broad and does not specify domain, language, method, or review intent
    Output:
    {{"question":"Você quer uma visão geral sobre análise de sentimentos ou quer focar em algo mais específico, como redes sociais, avaliações de produtos, notícias, português, modelos baseados em transformers ou métricas de avaliação?","reason":"sentiment analysis can vary by domain, language, method, and evaluation focus"}}

    User query: federated learning in healthcare
    Reason from previous assessment: the query specifies method and domain, but not the specific research focus
    Output:
    {{"question":"Em federated learning na saúde, você quer uma visão geral da área ou quer focar em privacidade, imagens médicas, prontuários eletrônicos, heterogeneidade dos dados, segurança ou aplicações clínicas?","reason":"the method and domain are clear, but the research focus is still open"}}

    Now generate the final answer for the current user query only.

    Required JSON shape:
    {{"question":"one clarification question with topic-specific options","reason":"short reason"}}
    """

    return PromptSpec(
        name="context_question",
        version=CONTEXT_QUESTION_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "query_context_question",
        },
    )


def build_rewrite_user_query_prompt(
    initial_query: str,
    question: str,
    question_reason: str,
    user_answer: str,
) -> PromptSpec:
    """Prompt for proposing one refined user topic after clarification."""

    text = f"""Return only one JSON object. No markdown. No explanation.

You are preparing an academic paper search in Computer Science or computationally-oriented research.

Task:
Create one clearer academic search topic from the initial query and the user's clarification answer.

This output is not a list of search queries.
It is one refined user topic that will later be passed to academic query planning.

Input language:
The initial query, clarification question, and user answer may be in Portuguese, English, or mixed language.

Output language:

* proposed_query must be in English because it will be used for academic paper search.
* message must be in Portuguese because it is shown to the user.
* reason must be in English because it is internal.
* JSON keys must stay in English.

Main goal:
Combine the initial query with the user's clarification answer.
Use only information actually provided by the user.
Do not invent methods, datasets, metrics, domains, applications, modalities, or restrictions.

Core rules:

* Create exactly one refined academic topic.
* Do not create multiple search queries.
* Preserve the main topic from the initial query.
* Preserve useful technical terms from the initial query.
* Use the clarification question only to understand what the user's answer refers to.
* Translate the user's intended meaning into English for proposed_query.
* Keep proposed_query short, clear, and searchable.
* Do not make the topic more specific than the user's answer allows.
* If the user's answer adds a specific focus, include that focus.
* If the user's answer is broad or general, keep the refined topic broad.
* If the user selects one option from the question, include only that selected option.
* If the user rejects an option, do not include it.
* If the user's answer is ambiguous but indicates general interest, produce a review-oriented topic.

General/overview intent:
If the user says something equivalent to:

* "geral"
* "visão geral"
* "quero entender a área"
* "introdução"
* "revisão"
* "survey"
* "estado da arte"
* "panorama"
* "não sei ainda"
* "quero algo mais amplo"

Then produce a review-oriented proposed_query using expressions such as:

* review of [topic]
* survey on [topic]
* overview of [topic]
* state of the art in [topic]

Do not add a specific method when the user asks for a general overview.

Computer Science scope:
Prefer refined topics that are clearly useful for academic search in Computer Science or computational research, such as:

* artificial intelligence
* machine learning
* deep learning
* natural language processing
* computer vision
* audio processing
* signal processing
* information retrieval
* recommendation systems
* cybersecurity
* software engineering
* databases
* distributed systems
* human-computer interaction
* data science
* graph learning
* time series forecasting
* optimization
* robotics
* computational healthcare
* educational technology
* computational finance
* IoT and sensor systems

Do not force the topic into one of these areas unless the initial query or user answer supports it.

Important negative rules:

* Do not add "deep learning" unless the user mentioned it or the initial query already implies it.
* Do not add "machine learning" unless the topic clearly involves computational modeling or the user selected it.
* Do not add a dataset unless the user mentioned it.
* Do not add evaluation metrics unless the user mentioned evaluation or metrics.
* Do not add a modality such as image, video, audio, text, graph, sensor, or time series unless it appears in the initial query, clarification question, or user answer.
* Do not add an application domain such as healthcare, finance, education, agriculture, or cybersecurity unless it appears in the initial query or user answer.
* Do not transform a broad answer into a narrow method.
* Do not transform a domain answer into a method answer.
* Do not include unrelated options from examples.

Good behavior examples:

Initial query: detecção de violência
Clarification question: Você quer uma visão geral sobre detecção de violência ou quer focar em uma modalidade específica, como áudio, vídeo, texto ou imagens?
Why the question matters: the modality changes the academic search terms
User answer: áudio
Output:
{{"proposed_query":"audio-based violence detection","message":"Com base na sua resposta, a busca ficaria: audio-based violence detection","reason":"the user specified audio as the modality"}}

Initial query: detecção de violência por áudio
Clarification question: Você quer uma visão geral sobre detecção de violência por áudio ou quer focar em datasets, modelos, extração de características, métricas ou detecção em tempo real?
Why the question matters: the topic has a modality, but the research focus is still open
User answer: visão geral
Output:
{{"proposed_query":"audio-based violence detection review","message":"Com base na sua resposta, a busca ficaria: audio-based violence detection review","reason":"the user asked for a general overview and the initial query already specifies audio-based violence detection"}}

Initial query: previsão de ações
Clarification question: Você quer uma visão geral sobre previsão de ações ou quer focar em métodos, fontes de dados, horizonte temporal ou métricas?
Why the question matters: different research focuses lead to different academic search terms
User answer: quero usar LSTM com notícias e sentimento de mercado
Output:
{{"proposed_query":"stock price prediction using LSTM, news, and market sentiment","message":"Com base na sua resposta, a busca ficaria: stock price prediction using LSTM, news, and market sentiment","reason":"the user specified the method and data sources"}}

Initial query: dados ruidosos em classificação de imagens médicas
Clarification question: Você quer focar em rótulos ruidosos, ruído na imagem, ruído de aquisição, robustez do modelo ou uma revisão geral do tema?
Why the question matters: different meanings of noise lead to different academic search terms
User answer: rótulos ruidosos
Output:
{{"proposed_query":"label noise in medical image classification","message":"Com base na sua resposta, a busca ficaria: label noise in medical image classification","reason":"the user specified label noise as the focus"}}

Initial query: engenharia de prompt
Clarification question: Você quer uma visão geral sobre engenharia de prompt ou quer focar em LLMs, geração de imagens, geração de código, avaliação de prompts ou otimização?
Why the question matters: prompt engineering can involve different applications and research focuses
User answer: avaliação de prompts em LLMs
Output:
{{"proposed_query":"prompt evaluation for large language models","message":"Com base na sua resposta, a busca ficaria: prompt evaluation for large language models","reason":"the user specified prompt evaluation and LLMs as the focus"}}

Initial query: recommendation systems
Clarification question: Você quer uma visão geral sobre sistemas de recomendação ou quer focar em filtragem colaborativa, deep learning, cold start, fairness ou explicabilidade?
Why the question matters: recommendation systems can be studied through different methods and evaluation concerns
User answer: cold start
Output:
{{"proposed_query":"cold start problem in recommendation systems","message":"Com base na sua resposta, a busca ficaria: cold start problem in recommendation systems","reason":"the user specified cold start as the research focus"}}

Initial query: segurança em APIs
Clarification question: Você quer uma visão geral sobre segurança em APIs ou quer focar em autenticação, autorização, vulnerabilidades, testes automatizados ou segurança em microsserviços?
Why the question matters: API security can involve different technical concerns
User answer: vulnerabilidades em REST APIs
Output:
{{"proposed_query":"security vulnerabilities in REST APIs","message":"Com base na sua resposta, a busca ficaria: security vulnerabilities in REST APIs","reason":"the user specified vulnerabilities in REST APIs"}}

Initial query: detecção de malware
Clarification question: Você quer uma visão geral sobre detecção de malware ou quer focar em análise estática, análise dinâmica, deep learning, Android malware ou datasets?
Why the question matters: malware detection can vary by platform, method, and data source
User answer: Android
Output:
{{"proposed_query":"Android malware detection","message":"Com base na sua resposta, a busca ficaria: Android malware detection","reason":"the user specified Android as the platform"}}

Initial query: aprendizado federado na saúde
Clarification question: Você quer uma visão geral sobre aprendizado federado na saúde ou quer focar em privacidade, imagens médicas, prontuários eletrônicos, heterogeneidade dos dados ou segurança?
Why the question matters: federated learning in healthcare can involve different technical and application focuses
User answer: privacidade
Output:
{{"proposed_query":"privacy in federated learning for healthcare","message":"Com base na sua resposta, a busca ficaria: privacy in federated learning for healthcare","reason":"the user specified privacy as the focus"}}

Initial query: sistemas distribuídos
Clarification question: Você quer uma visão geral sobre sistemas distribuídos ou quer focar em consenso, tolerância a falhas, escalabilidade, microsserviços ou computação em nuvem?
Why the question matters: distributed systems is broad and the selected focus changes the search terms
User answer: consenso
Output:
{{"proposed_query":"consensus algorithms in distributed systems","message":"Com base na sua resposta, a busca ficaria: consensus algorithms in distributed systems","reason":"the user specified consensus as the focus"}}

Initial query: bancos de dados vetoriais
Clarification question: Você quer uma visão geral sobre bancos de dados vetoriais ou quer focar em busca aproximada, indexação, recuperação semântica, RAG ou comparação de desempenho?
Why the question matters: vector databases can be studied through indexing, retrieval, systems, or applications
User answer: RAG
Output:
{{"proposed_query":"vector databases for retrieval-augmented generation","message":"Com base na sua resposta, a busca ficaria: vector databases for retrieval-augmented generation","reason":"the user specified RAG as the application context"}}

Initial query: graph neural networks
Clarification question: Você quer uma visão geral sobre graph neural networks ou quer focar em classificação de nós, predição de links, grafos dinâmicos, explicabilidade ou aplicações?
Why the question matters: graph neural networks have different tasks and application directions
User answer: link prediction
Output:
{{"proposed_query":"link prediction using graph neural networks","message":"Com base na sua resposta, a busca ficaria: link prediction using graph neural networks","reason":"the user specified link prediction as the task"}}

Initial query: UX em aplicativos educacionais
Clarification question: Você quer uma visão geral sobre UX em aplicativos educacionais ou quer focar em usabilidade, acessibilidade, engajamento, avaliação com usuários ou design centrado no usuário?
Why the question matters: HCI and UX research can vary by evaluation focus
User answer: avaliação com usuários
Output:
{{"proposed_query":"user evaluation of educational applications in human-computer interaction","message":"Com base na sua resposta, a busca ficaria: user evaluation of educational applications in human-computer interaction","reason":"the user specified user evaluation in the context of educational applications"}}

Initial query: séries temporais
Clarification question: Você quer uma visão geral sobre séries temporais ou quer focar em previsão, classificação, detecção de anomalias, transformers ou aplicações específicas?
Why the question matters: time series research can involve different tasks and methods
User answer: detecção de anomalias
Output:
{{"proposed_query":"time series anomaly detection","message":"Com base na sua resposta, a busca ficaria: time series anomaly detection","reason":"the user specified anomaly detection as the task"}}

Initial query: classificação de sentimentos
Clarification question: Você quer uma visão geral sobre análise de sentimentos ou quer focar em redes sociais, avaliações de produtos, português, transformers ou métricas de avaliação?
Why the question matters: sentiment analysis can vary by domain, language, method, and evaluation focus
User answer: português
Output:
{{"proposed_query":"sentiment analysis in Portuguese","message":"Com base na sua resposta, a busca ficaria: sentiment analysis in Portuguese","reason":"the user specified Portuguese as the language"}}

Initial query: LLM hallucination
Clarification question: Você quer uma visão geral sobre alucinação em LLMs ou quer focar em causas, métricas de avaliação, mitigação, RAG, verificação factual ou benchmarks?
Why the question matters: LLM hallucination can be studied through causes, evaluation, mitigation, and system design
User answer: mitigação com RAG
Output:
{{"proposed_query":"mitigating hallucination in large language models using retrieval-augmented generation","message":"Com base na sua resposta, a busca ficaria: mitigating hallucination in large language models using retrieval-augmented generation","reason":"the user specified mitigation with RAG"}}

Initial query:
{initial_query}

Clarification question:
{question}

Why the question matters:
{question_reason}

User answer:
{user_answer}

Required JSON shape:
{{"proposed_query":"clear academic search topic","message":"Com base na sua resposta, a busca ficaria: ...","reason":"short reason"}}

"""

    return PromptSpec(
        name="rewrite_user_query",
        version=REWRITE_USER_QUERY_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "clarified_user_query_rewrite",
        },
    )


def build_rewrite_from_user_revision_prompt(
    initial_query: str,
    proposed_query: str,
    clarification_question: str,
    clarification_reason: str,
    clarification_answer: str,
    user_revision: str,
) -> PromptSpec:
    """Prompt for refining a user-written revision of the proposed topic."""

    text = f"""Return only one JSON object. No markdown. No explanation.
                Create one clear academic search topic from the user's revision.

                Rules:
                - Prefer the revision written by the user.
                - Preserve useful context from the initial query and clarification.
                - Do not invent methods, datasets, domains, or restrictions.
                - Do not create multiple queries.
                - This is still the refined user topic, not generated academic
                  search queries.
                - The proposed_query must be in English because it will be used for academic paper search.
                - The message must be in Portuguese because it is shown to the user.
                - The reason must be in English because it is internal.
                - Keep JSON keys in English.

                Initial query: {initial_query}
                Previous proposed query: {proposed_query}
                Clarification question: {clarification_question}
                Clarification reason: {clarification_reason}
                Clarification answer: {clarification_answer}
                User revision: {user_revision}

                Exact shape:
                {{"proposed_query":"clear academic search topic","message":"Com base na sua revisão, a busca ficaria: ...","reason":"short reason"}}
            """

    return PromptSpec(
        name="rewrite_from_user_revision",
        version=REWRITE_FROM_USER_REVISION_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "user_revision_query_rewrite",
        },
    )

def build_initial_queries_prompt(context: SearchContext) -> PromptSpec:
    """Prompt for initial exploratory search queries."""

    text = f"""Return only one JSON object. No markdown. No explanation.

You are planning the first round of academic paper search queries.

User topic:
{context.user_query}

Minimum year:
{context.min_year}

Task:
Create 3 to 5 short exploratory academic search queries for finding papers.

This is the first search round.
The goal is broad but controlled exploration.
Do not make the queries too specific too early.

The user topic was already assessed or refined before reaching this step.
Do not ask questions here.
Do not reassess whether the topic has enough context.

Core goal:
Generate simple search queries that are faithful to the user's topic.
The first round should cover the main topic, close synonyms, and obvious academic variants.
Do not add search intentions that the user did not express.

Initial search strategy:
- Query 1 should be the closest academic version of the user topic.
- Query 2 should use a close synonym or equivalent academic wording.
- Query 3 should use another common term for the same task/problem.
- Query 4 may add a very broad computational or application context only if it is naturally implied.
- Query 5 may add review/survey intent only if the user explicitly asked for overview, review, survey, state of the art, or general understanding.

Important:
The first round should avoid overfitting.
Do not start with highly specific datasets, metrics, architectures, benchmarks, or narrow subproblems unless the user mentioned them.

Before writing the queries, analyze internally:
- What is the user's main topic?
- What task, method, modality, domain, application, or problem is explicitly present?
- What are the closest academic synonyms?
- What terms would retrieve relevant Computer Science or computational papers?
- What terms would cause topic drift?
- What should not be added because the user did not mention it?

Important rules:
- Return only the final JSON object.
- Do not include your internal analysis.
- The "queries" array must contain 3 to 5 non-empty strings.
- Academic search queries must be in English, even if the user topic is in Portuguese.
- Translate only the meaning needed for academic search.
- The reason must be in English because it is internal.
- Every query must preserve the user's main intent.
- Keep each query short and searchable.
- Prefer precise academic terms.
- Prefer Computer Science or computational wording when faithful to the topic.
- Do not invent datasets, methods, metrics, years, restrictions, or narrow applications.
- Do not add "review", "survey", "overview", "state of the art", "taxonomy", or "systematic review" unless the user explicitly asked for review/overview/general understanding.
- Do not add "deep learning", "machine learning", "benchmark", "datasets", or "evaluation" unless stated or clearly implied by the topic.
- Do not broaden a specific topic into an unrelated generic field.
- Do not narrow a broad topic into one specific method unless the method was mentioned.

Review intent rules:
Only use review-oriented terms when the user topic contains words such as:
review, survey, overview, literature review, systematic review, state of the art,
general overview, introduction, panorama, visão geral, revisão, estado da arte,
or equivalent expressions for overview, review, or general understanding.

If the user did not ask for review, do not add review/survey terms.

Modality rules:
- If the topic contains a modality, preserve that modality in every query when possible.
- Explicit modalities include audio, speech, sound, acoustic signal, image, video, text, graph, time series, sensor data, tabular data, network traffic, source code, logs, and multimodal data.
- If the topic does not contain an explicit modality, do not invent one.
- Do not turn one modality into another.
- If the topic contains a method, domain, task, dataset type, restriction, or application, preserve it.
- If a modality is required, use close variants of the same modality, not unrelated modalities.

First-round behavior:
Good first-round queries are broad and close to the topic.
Bad first-round queries are too narrow, too technical, or based on assumptions.

Bad behavior:
- Do not add a specific model if the user did not mention it.
- Do not add a specific dataset if the user did not mention it.
- Do not add a specific metric if the user did not mention it.
- Do not add a specific domain if the user did not mention it.
- Do not add multimodal, deep learning, benchmark, dataset, or evaluation terms unless stated or clearly implied.
- Do not add review/survey if the user only described the topic.
- Do not copy examples unless they fit the current topic.

Examples:

User topic: audio violence detection
Output:
{{"queries":[
  "audio violence detection",
  "audio-based violence detection",
  "audio-only violence detection",
  "acoustic violence detection",
  "sound-based violence detection"
],"reason":"the first round explores close audio-based variants of the same violence detection topic without adding specific methods or datasets"}}

User topic: audio violence detection overview
Output:
{{"queries":[
  "audio violence detection review",
  "audio-based violence detection survey",
  "audio-only violence detection overview",
  "acoustic violence detection methods",
  "sound-based violence detection state of the art"
],"reason":"the user requested an overview, so the first round includes review-oriented variants while preserving the audio modality"}}

User topic: plant disease detection
Output:
{{"queries":[
  "plant disease detection",
  "automatic plant disease detection",
  "computational plant disease detection",
  "plant disease classification",
  "plant disease recognition"
],"reason":"the first round explores close variants of the plant disease detection task without inventing a modality or method"}}

User topic: image-based plant disease detection
Output:
{{"queries":[
  "image-based plant disease detection",
  "plant disease detection using images",
  "plant disease image classification",
  "leaf image disease detection",
  "computer vision plant disease detection"
],"reason":"the topic specifies image-based detection, so all queries preserve the image/computer vision direction"}}

User topic: graph neural networks for fraud detection
Output:
{{"queries":[
  "graph neural networks for fraud detection",
  "GNN fraud detection",
  "graph-based fraud detection using neural networks",
  "fraud detection with graph learning",
  "graph representation learning for fraud detection"
],"reason":"the topic specifies graph neural networks and fraud detection, so the first round explores close graph-learning variants"}}

User topic: cybersecurity intrusion detection
Output:
{{"queries":[
  "cybersecurity intrusion detection",
  "network intrusion detection",
  "intrusion detection systems",
  "attack detection systems",
  "computational intrusion detection"
],"reason":"the topic specifies a cybersecurity detection task, so the first round explores common intrusion detection terminology"}}

User topic: stock price prediction using LSTM and news sentiment
Output:
{{"queries":[
  "stock price prediction using LSTM and news sentiment",
  "stock market forecasting with LSTM and sentiment analysis",
  "financial time series forecasting using news sentiment",
  "stock prediction using recurrent neural networks and sentiment",
  "news sentiment stock price forecasting"
],"reason":"the topic specifies the task, method, and data source, so the first round preserves all three"}}

User topic: noisy labels in medical image classification
Output:
{{"queries":[
  "noisy labels in medical image classification",
  "medical image classification with noisy labels",
  "label noise in medical imaging",
  "robust medical image classification under label noise",
  "learning with noisy labels in medical image classification"
],"reason":"the topic specifies label noise and medical image classification, so the first round explores close robust-learning variants"}}

User topic: prompt engineering for image generation
Output:
{{"queries":[
  "prompt engineering for image generation",
  "prompt design for text-to-image generation",
  "text-to-image prompt engineering",
  "prompt optimization for image generation",
  "prompting techniques for text-to-image models"
],"reason":"the topic specifies prompt engineering for image generation, so the first round uses close text-to-image variants"}}

User topic: LLM hallucination mitigation with RAG
Output:
{{"queries":[
  "LLM hallucination mitigation with RAG",
  "mitigating hallucinations in large language models using retrieval-augmented generation",
  "retrieval-augmented generation for hallucination reduction",
  "RAG for factuality in large language models",
  "reducing hallucinations in LLMs with retrieval"
],"reason":"the topic specifies LLM hallucination mitigation and RAG, so the first round preserves both"}}

Exact shape:
{{"queries":["academic search query 1","academic search query 2","academic search query 3"],"reason":"short reason"}}
"""

    return PromptSpec(
        name="initial_queries",
        version=INITIAL_QUERIES_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "initial_query_planning",
        },
    )

def build_plan_filters_prompt(context: SearchContext) -> PromptSpec:
    """Prompt for planning semantic validation filters."""

    text = f"""Return only one JSON object. No markdown. No explanation.

You are planning semantic filters for academic paper validation.

User refined topic:
{context.user_query}

Task:
Extract explicit semantic validation criteria from the refined topic.
These filters will be used internally before validating candidate papers.

Language rule:
All filter values, reasons, rules, concepts, and examples must be in English.
Keep JSON keys in English.

Core goal:
Separate what is mandatory from what is only useful or preferred.
Do not generate search queries here.
Do not validate papers here.

Rules:
- Extract criteria only from the refined topic.
- Do not invent a focus that the user did not provide.
- Do not add datasets, methods, metrics, domains, modalities, or applications unless they are stated or clearly implied.
- The primary_intent must describe the main academic task or problem.
- required_concepts must list concepts that must be present for a paper to be relevant.
- required_modality must list required input modality terms only when the topic explicitly implies a modality.
- Explicit modalities include audio, speech, sound, acoustic signal, image, video, text, graph, time series, sensor data, tabular data, network traffic, source code, logs, and multimodal data.
- If the refined topic does not contain an explicit modality, required_modality must be an empty array.
- positive_signals must list terms or concepts that are good evidence of relevance.
- negative_signals must list terms or concepts that indicate mismatch.
- hard_exclusion_rules must be strict rules that force exclusion during validation.
- soft_preferences must list useful but non-mandatory aspects.
- validation_priority must be an ordered list of what matters most when validating papers.
- Use an empty array for any list field that has no applicable items.

Mandatory topic decomposition:
- Identify the central task, the required modality, and the target phenomenon.
- If the refined topic contains a modality, that modality must become a required modality.
- If the refined topic contains a detection/classification task, the target phenomenon must become a required concept.
- For a paper to be relevant, it must satisfy both:
  1. the central task or problem;
  2. the required modality, when a modality is present.
- Do not treat broad related areas as relevant if they miss the required modality or target phenomenon.

Required modality interpretation:
- required_modality should contain acceptable equivalent terms for the same modality.
- A paper satisfies required_modality when it clearly matches at least one of these terms.
- Example: for audio-based topics, acceptable terms may include audio, sound, speech, acoustic signal, acoustic event, or audio signal.
- A paper about video, image, text, social media, or general multimodal analysis must not be considered relevant unless the required modality is central.

Required concept interpretation:
- required_concepts should contain mandatory topic concepts, not loose keywords.
- Prefer phrases such as "violence detection", "fake news detection", "plant disease detection", or "label noise".
- Do not split a mandatory concept into vague words if that would allow false positives.

Computer Science validation guidance:
Plan filters so validation prefers papers with clear Computer Science or strongly computational contributions.
A paper should be relevant only when there is clear evidence in the title, abstract, keywords, or venue that it includes a central computational contribution.

Positive computational signals include:
- algorithms, models, computational methods, systems, architectures, pipelines, frameworks, software, implementation, or computational tools;
- areas such as Artificial Intelligence, Machine Learning, Deep Learning, Data Science, Information Retrieval, NLP, Computer Vision, Audio Processing, Signal Processing, Cybersecurity, Software Engineering, Databases, Distributed Systems, HCI, or related areas;
- empirical evaluation of computational techniques, including benchmarks, datasets, metrics, experiments, performance comparison, ablation studies, or system validation;
- application of computational methods in domains such as healthcare, education, law, finance, agriculture, industry, or social media, when the computational method is central.

General hard exclusions:
The hard_exclusion_rules field must include these general exclusions when they apply:
- Exclude papers without a clear Computer Science or computational contribution.
- Exclude papers that only share superficial keywords with the refined topic.
- Exclude papers from non-computational domains when computation is not central to the method, model, system, dataset, or technical evaluation.
- Exclude conceptual, legal, ethical, social, or historical papers without a concrete computational contribution.
- Exclude papers that mention technology only casually.
- Exclude papers that do not match the user's primary search intent.

Example:
User refined topic: audio-based violence detection review
Output:
{{
  "primary_intent": "computational audio-based violence detection",
  "required_concepts": ["violence detection"],
  "required_modality": ["audio", "sound", "speech", "acoustic signal", "audio signal"],
  "positive_signals": [
    "audio-based violence detection",
    "audio signal",
    "acoustic event detection",
    "violent event detection",
    "aggression detection",
    "audio classification",
    "signal processing",
    "machine learning model",
    "deep learning model",
    "dataset",
    "benchmark",
    "experimental evaluation",
    "survey",
    "review"
  ],
  "negative_signals": [
    "plant disease detection",
    "hate speech detection",
    "deepfake detection",
    "video-only violence detection",
    "image-only violence detection",
    "visual surveillance only",
    "intimate partner violence prevalence",
    "domestic violence healthcare disclosure",
    "gender-based violence policy",
    "social science review",
    "medical meta-analysis"
  ],
  "hard_exclusion_rules": [
    "Exclude if the paper is not about violence detection or violent/aggressive event detection.",
    "Exclude if the paper does not use audio, sound, speech, acoustic signal, or audio signal as a central input modality.",
    "Exclude if the paper is mainly about plant disease, hate speech, deepfakes, healthcare disclosure, prevalence, policy, sociology, or non-computational violence studies.",
    "Exclude if the paper is about video-only, image-only, or visual-only violence detection.",
    "Exclude if the paper only shares the word violence but does not study computational violence detection.",
    "Exclude if the paper has no clear Computer Science or computational contribution."
  ],
  "soft_preferences": [
    "review or survey papers",
    "audio-only methods",
    "datasets",
    "benchmarks",
    "feature extraction",
    "model comparison",
    "real-time detection",
    "computational efficiency"
  ],
  "validation_priority": [
    "required modality match",
    "violence detection task match",
    "computer science contribution",
    "hard exclusions",
    "review or survey usefulness",
    "research usefulness"
  ]
}}
"""

    return PromptSpec(
        name="plan_filters",
        version=PLAN_FILTERS_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "semantic_filter_planning",
        },
    )

def build_validate_papers_prompt(
    context: SearchContext,
    papers: list[Paper],
    search_feedback: dict[str, object] | None = None,
    search_filters: dict[str, object] | None = None,
) -> PromptSpec:
    """Prompt for semantic validation of candidate papers."""

    feedback = search_feedback or {}
    filters = search_filters or {}
    revised_topic = _feedback_text(feedback, "revised_topic") or context.user_query
    positive_constraints = _feedback_list(feedback, "positive_constraints")
    negative_constraints = _feedback_list(feedback, "negative_constraints")
    query_strategy = _feedback_text(feedback, "query_strategy") or "none"
    primary_intent = _feedback_text(filters, "primary_intent") or "none"
    required_concepts = _feedback_list(filters, "required_concepts")
    required_modality = _feedback_list(filters, "required_modality")
    positive_signals = _feedback_list(filters, "positive_signals")
    negative_signals = _feedback_list(filters, "negative_signals")
    hard_exclusion_rules = _feedback_list(filters, "hard_exclusion_rules")
    soft_preferences = _feedback_list(filters, "soft_preferences")
    validation_priority = _feedback_list(filters, "validation_priority")
    paper_block = _format_candidate_papers(papers)
    candidate_count = len(papers)
    candidate_label = "candidate paper" if candidate_count == 1 else "candidate papers"

    text = f"""Return only one JSON object. No markdown. No explanation.

You are validating the current batch of academic search results semantically.

The current batch contains exactly {candidate_count} {candidate_label}.

User search intent:
{context.user_query}

Current revised topic:
{revised_topic}

Positive constraints:
{_format_items(positive_constraints)}

Negative constraints:
{_format_items(negative_constraints)}

Query strategy:
{query_strategy}

Planned semantic filters:
Primary intent: {primary_intent}
Required concepts: {_format_items(required_concepts)}
Required modality: {_format_items(required_modality)}
Positive signals: {_format_items(positive_signals)}
Negative signals: {_format_items(negative_signals)}
Hard exclusion rules: {_format_items(hard_exclusion_rules)}
Soft preferences: {_format_items(soft_preferences)}
Validation priority: {_format_items(validation_priority)}

Current candidate batch:
{paper_block}

Task:
Validate only the {candidate_count} {candidate_label} in the current candidate batch.
For each candidate in this batch, decide whether it is relevant to the user's search intent.

Critical output rule:
You must return exactly {candidate_count} validation object(s) inside validated_papers.
Return exactly one validation object for each candidate in the current batch.
If the current batch contains one paper, validated_papers must still be a list with exactly one object.
Do not skip any candidate.
Do not add papers that are not in the current candidate batch.
The paper_id must exactly match the candidate paper id.
Do not use title, DOI, URL, index, or generated identifiers as paper_id.
Each validation object must be based only on that same candidate paper.

Completeness rule:
A summary alone is not enough.
The validated_papers list is mandatory.
Every candidate in the current batch must appear in validated_papers.
If evidence is weak or the abstract is missing, still return a validation object and judge cautiously.

For each paper, check:
1. Apply validation_priority in order.
2. Is there clear evidence of a Computer Science or strongly computational contribution?
3. Does it match the primary_intent?
4. Does it satisfy required_concepts?
5. Does it satisfy required_modality, if any?
6. Does it violate any hard_exclusion_rules, negative_signals, or negative constraints?
7. Would it actually help the user with this search?

Rules:
- Evaluate semantic relation to the user's intent, not superficial word overlap.
- Use the current revised topic as the main reference.
- Use the planned semantic filters as explicit validation criteria.
- Apply validation_priority before assigning high, medium, low, or reject.
- First verify clear evidence of a Computer Science or strongly computational contribution in the title, abstract, source, venue-like metadata, or available paper metadata.
- A computational contribution may be an algorithm, model, computational method, system, architecture, pipeline, framework, software, implementation, tool, dataset, benchmark, metric, experiment, performance comparison, ablation study, or system validation.
- Papers may apply computational methods to healthcare, education, law, finance, agriculture, industry, social media, or other domains, but the computational method must be central.
- Exclude papers that are mainly legal, political, sociological, philosophical, historical, medical, business, conceptual, ethical, or social unless they contain a concrete computational method, model, system, dataset, or technical evaluation.
- Exclude papers that mention technology only casually.
- The relevance_reason for included papers must cite concrete computational evidence from the title or abstract, such as a method, model, algorithm, system, dataset, task, or evaluation.
- Do not invent a computational contribution that is not supported by the title, abstract, or available metadata.
- Treat primary_intent as mandatory when it is not "none".
- Treat required_concepts, required_modality, and hard_exclusion_rules as mandatory criteria.
- Treat positive constraints as desired signals.
- Treat negative constraints as hard exclusions.
- Treat negative_signals as mismatch indicators.
- Exclude a paper if it violates any hard_exclusion_rule, even if it contains similar keywords.
- Write relevance_reason, mismatch_reason, useful_for, summary, and any internal reason text in English.
- Keep JSON keys in English.
- Keep the explanation short and based only on title, abstract, year, source, and URL.
- Do not invent information absent from the title or abstract.
- If the abstract is missing, judge cautiously using only title and metadata.
- If the paper title/abstract is about a different task, domain, or modality, exclude it.
- If the paper only shares an isolated term from the query but studies another meaning, domain, or non-computational topic, exclude it.
- Do not include a paper only because it matches one positive signal while failing a required concept, required modality, or hard exclusion rule.

Mandatory inclusion gate:
Before assigning high, medium, or low, check all mandatory gates.

A paper can be included only if:
1. It matches the primary_intent when primary_intent is not "none";
2. It satisfies the required_concepts;
3. It satisfies the required_modality when required_modality is not empty;
4. It does not violate any hard_exclusion_rule;
5. It has a clear Computer Science or computational contribution.

If any mandatory gate fails, the paper must be:
relevance="reject"
decision="exclude"

Required modality gate:
When required_modality is not empty, the paper must clearly use the required modality as a central input, data type, or analysis target.
Mentioning the modality casually is not enough.
If the paper uses another modality instead, exclude it.
If the paper is multimodal, include it only when the required modality is central to the method or evaluation.

Do not use high, medium, or low for papers that fail a mandatory gate.

Relevance labels:
- high: directly related to the revised topic and useful for the user's intent.
- medium: partially related and useful, but missing one secondary aspect.
- low: tangential; only useful as background.
- reject: wrong topic, wrong domain, wrong modality, wrong task, or violates constraints.

Decision labels:
- include: use for high and medium papers.
- include: use for low only if it is clearly useful background.
- exclude: use for reject.
- exclude: use for any paper that clearly violates negative constraints.
- exclude: use for any paper that fails required_concepts, required_modality, or hard_exclusion_rules.

Computer Science validation guidance:
1. Check for a clear computational contribution.
   If there is no algorithm, model, method, system, software, dataset, benchmark, experiment, or technical evaluation, exclude the paper.
2. Check primary_intent.
   If the paper has a computational contribution but solves a different main task or problem, exclude it.
3. Check required_concepts.
   If a mandatory concept is missing or used with a different meaning, exclude it.
4. Check required_modality only when required_modality is not empty.
   If the required modality is missing or only mentioned casually, exclude it.
5. Apply hard_exclusion_rules, negative_signals, and negative constraints.
   If any strict exclusion applies, exclude the paper.
6. Only after those checks, classify the paper as high, medium, or low.

General examples:

User search intent: fake news detection overview
Current revised topic: fake news detection review
Candidate title: A Survey on Fake News Detection: Methods, Datasets, and Evaluation
Expected:
{{"relevance":"high","decision":"include","relevance_reason":"The paper is a survey about fake news detection methods, datasets, and evaluation.","mismatch_reason":"","useful_for":"overview"}}

User search intent: fake news detection overview
Current revised topic: fake news detection review
Candidate title: Fake News and Political Polarization: A Sociological Essay
Expected:
{{"relevance":"low","decision":"exclude","relevance_reason":"The paper discusses fake news, but not fake news detection methods.","mismatch_reason":"It focuses on sociological effects rather than detection.","useful_for":"not useful"}}

User search intent: stock price prediction using LSTM and news sentiment
Current revised topic: stock price prediction using LSTM and news sentiment
Candidate title: Stock Price Forecasting Using LSTM Networks and Financial News Sentiment
Expected:
{{"relevance":"high","decision":"include","relevance_reason":"The paper matches the task, method, and data source.","mismatch_reason":"","useful_for":"methods and evaluation"}}

User search intent: stock price prediction using LSTM and news sentiment
Current revised topic: stock price prediction using LSTM and news sentiment
Candidate title: Weather Forecasting Using LSTM and Sentiment Analysis
Expected:
{{"relevance":"reject","decision":"exclude","relevance_reason":"","mismatch_reason":"The paper uses similar methods but is about weather forecasting, not stock prediction.","useful_for":"not useful"}}

User search intent: noisy labels in medical image classification
Current revised topic: label noise in medical image classification
Candidate title: Robust Learning with Noisy Labels for Medical Image Classification
Expected:
{{"relevance":"high","decision":"include","relevance_reason":"The paper directly addresses noisy labels in medical image classification.","mismatch_reason":"","useful_for":"methods and robustness"}}

User search intent: noisy labels in medical image classification
Current revised topic: label noise in medical image classification
Candidate title: Denoising MRI Images under Acquisition Noise
Expected:
{{"relevance":"low","decision":"exclude","relevance_reason":"The paper is related to medical image noise, but not label noise in classification.","mismatch_reason":"It focuses on acquisition/image noise rather than noisy labels.","useful_for":"not useful"}}

User search intent: prompt engineering for image generation
Current revised topic: prompt engineering for image generation
Candidate title: Prompt Engineering for Text-to-Image Generation Models
Expected:
{{"relevance":"high","decision":"include","relevance_reason":"The paper directly matches prompt engineering for image generation.","mismatch_reason":"","useful_for":"methods"}}

User search intent: prompt engineering for image generation
Current revised topic: prompt engineering for image generation
Candidate title: Prompt Engineering for Code Generation with Large Language Models
Expected:
{{"relevance":"reject","decision":"exclude","relevance_reason":"","mismatch_reason":"The paper is about code generation, not image generation.","useful_for":"not useful"}}

Special rule for overview/review searches:
If the user explicitly asked for overview, review, survey, introduction, or state of the art:
- include surveys, reviews, taxonomies, benchmarks, and broad comparative papers that match the topic;
- include representative method papers as medium if they help map the area;
- reject narrow papers from a different domain, modality, or task.

Special rule for strict constraints:
If negative constraints are provided, reject papers that clearly violate them.
If positive constraints are provided, prefer papers that explicitly match them.
If a paper is relevant to the broad topic but violates a constraint, exclude it.

Required JSON shape:
{{"validated_papers":[{{"paper_id":"paper id","relevance":"high","decision":"include","relevance_reason":"short reason","mismatch_reason":"","useful_for":"how this helps the research"}}],"summary":"short summary of relevant and rejected paper types"}}
"""

    return PromptSpec(
        name="validate_papers",
        version=VALIDATE_PAPERS_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "paper_semantic_validation",
        },
    )

def build_refine_queries_prompt(
    context: SearchContext,
    validated_papers: list[dict[str, object]],
    used_queries: list[str],
    search_feedback: dict[str, object] | None = None,
) -> PromptSpec:
    """Prompt for refining search queries after feedback and validation."""

    evidence = _format_validated_papers(validated_papers[:10])
    used = "; ".join(used_queries[-8:]) or "none"
    feedback = search_feedback or {}
    revised_topic = _feedback_text(feedback, "revised_topic") or context.user_query
    positive_constraints = _feedback_list(feedback, "positive_constraints")
    negative_constraints = _feedback_list(feedback, "negative_constraints")
    query_strategy = _feedback_text(feedback, "query_strategy") or "none"

    text = f"""Return only one JSON object. No markdown. No explanation.

You are planning a new round of academic search queries after previous search results were validated.

This is not the first search round.
Your job is to refine the search direction using:
1. the revised topic;
2. the user's feedback;
3. the positive constraints;
4. the negative constraints;
5. the previous validated-paper evidence;
6. the used queries.

User original topic:
{context.user_query}

Current revised topic:
{revised_topic}

Minimum year:
{context.min_year}

Used queries:
{used}

Validated-paper evidence:
{evidence}

Positive constraints:
{_format_items(positive_constraints)}

Negative constraints:
{_format_items(negative_constraints)}

Query strategy:
{query_strategy}

Task:
Create 1 to 3 new academic search queries.
The new queries must improve the search based on what worked and what failed.
Do not repeat used queries.

Core refinement strategy:
- If previous results were too broad, make the new queries more specific.
- If previous results matched the wrong domain, add terms that enforce the correct domain.
- If previous results matched the wrong modality, add terms that enforce the correct modality.
- If previous results matched the wrong task, add terms that enforce the correct task.
- If previous results matched the wrong method, avoid that method and use the intended one.
- If previous results were mostly irrelevant due to keyword ambiguity, use more precise academic terminology.
- If previous results were useful but narrow, explore a close synonym or adjacent wording.
- If the user gave feedback, follow it more strongly than the previous evidence.
- If positive constraints exist, include at least one of them in every query when natural.
- If negative constraints exist, avoid those terms and avoid queries likely to retrieve them.

Rules:
- Return only the final JSON object.
- The "queries" array must contain 1 to 3 non-empty strings.
- Academic search queries must be in English, even if the user topic is in Portuguese.
- Translate only the meaning needed for academic search.
- The reason must be in English because it is internal.
- Every query must follow the revised topic.
- Every query must respect positive constraints, negative constraints, and query strategy.
- Treat negative constraints as hard exclusions.
- Preserve any modality, method, domain, task, dataset type, application, or restriction that appears in the revised topic or constraints.
- If no modality is present in the revised topic or constraints, do not invent one.
- Do not invent methods, datasets, metrics, domains, modalities, applications, or review intent.
- Prefer Computer Science or computational search terms when they are faithful to the revised topic.
- Avoid terms from negative constraints.
- Avoid broad queries that caused irrelevant results before.
- Do not generate a query that is only a minor word-order variation of a used query.
- Do not copy examples unless they fit the current topic.

How to use validated-paper evidence:
- Included high/medium papers indicate useful terminology.
- Rejected papers indicate terms, domains, modalities, or meanings to avoid.
- Do not use rejected paper titles as positive query inspiration.
- Use mismatch reasons to avoid the same mistake.
- If evidence is "none", rely on the revised topic and constraints.

Good refinement behavior examples:

Case:
Revised topic: audio-based violence detection
Used queries: audio violence detection; automatic violence detection
Negative constraints: video; visual; multimodal; hate speech
Validated evidence shows many video-surveillance papers.
Output:
{{"queries":[
  "audio-only violence detection",
  "acoustic aggression detection",
  "sound-based violent event detection"
],"reason":"the previous search drifted toward visual or multimodal violence detection, so the new queries enforce the audio modality"}}

Case:
Revised topic: plant disease detection
Used queries: plant disease detection; automatic plant disease detection
Positive constraints: image-based; leaf images
Negative constraints: sensor data; genomic analysis
Validated evidence shows many agriculture-domain papers without computer vision.
Output:
{{"queries":[
  "image-based plant disease detection",
  "leaf image disease classification",
  "computer vision plant disease recognition"
],"reason":"the user or evidence indicates that the intended direction is image-based detection, so the new queries enforce the visual modality"}}

Case:
Revised topic: stock price prediction using LSTM and news sentiment
Used queries: stock price prediction; financial forecasting
Negative constraints: cryptocurrency; portfolio optimization
Validated evidence shows generic finance forecasting papers.
Output:
{{"queries":[
  "stock price prediction using LSTM and news sentiment",
  "news sentiment stock forecasting recurrent neural networks",
  "financial time series prediction with LSTM and sentiment analysis"
],"reason":"the previous search was too broad, so the new queries enforce the method and data source"}}

Case:
Revised topic: prompt evaluation for large language models
Used queries: prompt engineering; LLM prompting
Positive constraints: evaluation; benchmark
Negative constraints: image generation; code generation
Validated evidence shows many prompt design papers without evaluation.
Output:
{{"queries":[
  "prompt evaluation for large language models",
  "benchmarking prompts in large language models",
  "evaluation methods for LLM prompting"
],"reason":"the previous search retrieved broad prompt engineering papers, so the new queries focus on evaluation"}}

Case:
Revised topic: label noise in medical image classification
Used queries: noisy data medical imaging; medical image noise
Negative constraints: acquisition noise; denoising; image enhancement
Validated evidence shows many image denoising papers.
Output:
{{"queries":[
  "label noise in medical image classification",
  "learning with noisy labels in medical imaging",
  "robust classification under label noise in medical images"
],"reason":"the previous search confused image noise with label noise, so the new queries enforce noisy labels"}}

Required JSON shape:
{{"queries":["academic search query 1","academic search query 2","academic search query 3"],"reason":"short reason"}}
"""

    return PromptSpec(
        name="refine_queries",
        version=REFINE_QUERIES_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "query_refinement",
        },
    )

def build_feedback_analysis_prompt(
    original_query: str,
    refined_query: str,
    user_feedback: str,
    validated_papers: list[dict[str, object]],
    used_queries: list[str],
) -> PromptSpec:
    """Prompt for interpreting human feedback on ranked papers."""

    papers = _format_validated_papers(validated_papers[:20])
    used = "; ".join(used_queries[-10:]) or "none"

    text = f"""Return only one JSON object. No markdown. No explanation.

You are improving an academic paper search after the user reviewed the ranked papers.

Task:
Interpret the user's critique into structured search guidance before generating new queries.

Inputs:
Original user query: {original_query}
Current refined query: {refined_query}
User feedback: {user_feedback}
Used queries: {used}
Validated-paper evidence:
{papers}

Rules:
- Do not generate search queries here.
- Do not copy the raw feedback into query_strategy.
- Convert the feedback into a cleaner academic revised_topic.
- Write revised_topic, positive_constraints, negative_constraints, query_strategy, and reason in English.
- The reason must be in English because it is internal.
- Keep JSON keys in English.
- Put desired concepts, modalities, and terms in positive_constraints.
- Put excluded concepts, modalities, and terms in negative_constraints.
- Make query_strategy an actionable sentence for the next query-planning step.
- Keep each constraint short and searchable.
- Preserve the original research intent unless the user explicitly changes it.
- If the user specifies a modality, method, domain, dataset type, or technical restriction, include it in positive_constraints.
- Use included and excluded papers as evidence for what worked and what failed.
- If the user excludes a modality, method, domain, paper type, or interpretation, include concise related terms in negative_constraints when relevant.
- When the user asks for Computer Science papers, make query_strategy focus on algorithms, models, computational methods, systems, datasets, benchmarks, or technical evaluation.

Expected case:
Original user query: fake news detection
Current refined query: fake news detection review
User feedback: I want Computer Science papers with detection models or datasets, not political essays or sociological discussions
Output:
{{"revised_topic":"computational fake news detection review","positive_constraints":["fake news detection models","machine learning","NLP","datasets","technical evaluation"],"negative_constraints":["political essay","sociology","legal analysis","conceptual discussion","non-computational review"],"query_strategy":"Search for Computer Science papers on fake news detection that present models, algorithms, datasets, benchmarks, or technical evaluation. Avoid political, sociological, legal, or conceptual papers without a computational contribution.","reason":"The user clarified that relevant papers should have a central computational contribution rather than only discussing fake news as a social or political topic."}}

Required JSON shape:
{{"revised_topic":"clean revised academic topic","positive_constraints":["constraint"],"negative_constraints":["constraint"],"query_strategy":"actionable query strategy","reason":"short reason"}}
"""

    return PromptSpec(
        name="feedback_analysis",
        version=FEEDBACK_ANALYSIS_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "search_feedback_analysis",
        },
    )


def build_continue_decision_prompt(
    context: SearchContext,
    round_number: int,
    validated_papers: list[dict[str, object]],
    last_new_paper_count: int,
    last_new_useful_count: int,
) -> PromptSpec:
    """Prompt for deciding whether another search round is useful."""

    evidence = _format_validated_papers(validated_papers[:10])

    text = f"""Return only one JSON object. No markdown. No explanation.
                Decide if another academic search round is useful.
                Topic: {context.user_query}
                Round: {round_number} of {context.max_rounds}
                New papers: {last_new_paper_count}
                New useful validated papers: {last_new_useful_count}
                Validated-paper evidence: {evidence}
                The "continue" value must be true or false.
                The reason must be in English because it is internal.
                Exact shape: {{"continue":true,"reason":"short reason"}}
            """

    return PromptSpec(
        name="continue_decision",
        version=CONTINUE_DECISION_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "continue_decision",
        },
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
    return "; ".join(items) if items else "none"


def _format_candidate_papers(papers: list[Paper]) -> str:
    if not papers:
        return "none"

    lines: list[str] = []
    for position, paper in enumerate(papers, start=1):
        abstract = " ".join((paper.abstract or "").split())[:500] or "no abstract"
        lines.append(
            f"{position}. id={paper.id}; title={paper.title}; "
            f"year={paper.year or 'unknown'}; source={paper.source}; "
            f"url={paper.url or 'none'}; abstract={abstract}"
        )
    return "\n".join(lines)


def _format_validated_papers(validated_papers: list[dict[str, object]]) -> str:
    if not validated_papers:
        return "none"

    lines: list[str] = []
    for position, item in enumerate(validated_papers, start=1):
        paper = item.get("paper")
        title = getattr(paper, "title", "unknown")
        paper_id = getattr(paper, "id", item.get("paper_id", "unknown"))
        relevance = _feedback_text(item, "relevance") or "unknown"
        decision = _feedback_text(item, "decision") or "unknown"
        relevance_reason = _feedback_text(item, "relevance_reason")
        mismatch_reason = _feedback_text(item, "mismatch_reason")
        lines.append(
            f"{position}. id={paper_id}; title={title}; relevance={relevance}; "
            f"decision={decision}; why_in={relevance_reason or 'none'}; "
            f"why_out={mismatch_reason or 'none'}"
        )
    return "\n".join(lines)
