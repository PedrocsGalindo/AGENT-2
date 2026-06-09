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


INITIAL_QUERIES_PROMPT_VERSION = "1.0.4"
REFINE_QUERIES_PROMPT_VERSION = "1.0.2"
FEEDBACK_ANALYSIS_PROMPT_VERSION = "1.0.0"
VALIDATE_PAPERS_PROMPT_VERSION = "1.0.0"
CONTINUE_DECISION_PROMPT_VERSION = "1.0.0"
ENRICH_QUERY_PROMPT_VERSION = "1.0.1"
ASSESS_QUERY_CONTEXT_PROMPT_VERSION = "1.0.0"
CONTEXT_QUESTION_PROMPT_VERSION = "1.0.0"
REWRITE_USER_QUERY_PROMPT_VERSION = "1.0.0"
REWRITE_FROM_USER_REVISION_PROMPT_VERSION = "1.0.0"


def build_assess_query_context_prompt(context: SearchContext) -> PromptSpec:
    """contexto suficiente ou nao"""

    text = f"""Return only one JSON object. No markdown. No explanation.

You are preparing an academic paper search.

Task:
Decide whether the user's query has enough context to start building useful academic search queries.

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

User query: violence detection
Output:
{{"has_enough_context":false,"reason":"the query is broad and does not specify whether the user wants a general overview or a specific focus such as modality, datasets, models, or deployment"}}

User query: violence detection review
Output:
{{"has_enough_context":true,"reason":"the user explicitly indicated review intent for the topic"}}

User query: audio violence detection
Output:
{{"has_enough_context":false,"reason":"the query specifies the modality, but not whether the user wants a general overview or a specific focus such as datasets, models, features, metrics, real-time detection, or comparison"}}

User query: audio violence detection review
Output:
{{"has_enough_context":true,"reason":"the user explicitly indicated review intent for audio-based violence detection"}}

User query: audio violence detection using deep learning
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
    Use the same language as the user when possible.
    If the query is in Portuguese, ask in Portuguese.
    If the query is in English, ask in English.
    If the system around the user is in Portuguese and the user answer is expected in Portuguese, Portuguese is acceptable.

    Topic guidance:
    These are not fixed templates.
    Use them only when they fit the current query.

    * For stock prediction, finance, market forecasting, or asset prediction:
    natural options include review/overview, forecasting methods, asset type, market, time horizon, data source, technical indicators, news sentiment, fundamentals, macroeconomic data, or evaluation metrics.

    * For violence detection:
    natural options include review/overview, modality, audio, video, text, images, datasets, models, feature extraction, real-time detection, surveillance, mobile deployment, benchmark comparison, or evaluation metrics.

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

    User query: violence detection
    Reason from previous assessment: the query is broad and does not specify whether the focus is a general review or a specific modality
    Output:
    {{"question":"Você quer uma visão geral sobre detecção de violência ou quer focar em uma modalidade específica, como áudio, vídeo, texto ou imagens?","reason":"the modality strongly changes the search terms for violence detection"}}

    User query: audio violence detection
    Reason from previous assessment: the query has a modality, but does not specify whether the user wants a general review or a specific research focus
    Output:
    {{"question":"Você quer uma visão geral sobre detecção de violência por áudio ou quer focar em algo mais específico, como datasets, modelos, extração de características, métricas de avaliação, detecção em tempo real ou comparação entre abordagens?","reason":"audio already defines the modality, so the question focuses on the missing research direction"}}

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

You are preparing an academic paper search.

Task:
Create one clearer academic search topic from the initial query and the user's clarification answer.

This output is not a list of academic search queries.
It is the refined user topic that will later be passed to initial query planning.

Main goal:
Extract only the information that the user actually provided.
Combine it with the initial query to create one clearer academic topic.

Rules:
- Create only one clearer academic topic.
- Do not create multiple queries.
- Do not invent information.
- Do not invent methods, datasets, metrics, domains, applications, or restrictions.
- Use only the initial query and the user's answer.
- Use the clarification question only to understand what the user's answer refers to.
- Preserve the main topic from the initial query.
- Preserve useful technical terms from the initial query.
- If the initial query is in English and contains academic terms, prefer keeping the proposed query in English.
- If the user answer is in Portuguese but the initial query is in English, translate only the clarification intent, not the whole topic unnecessarily.
- If the user asks for a general view, overview, review, survey, introduction, or to understand the area, make the proposed query review-oriented.
- If the user says "geral", "visão geral", "quero entender", "quero aprender", "overview", "review", or similar, do not add a specific method.
- If the user selected one option from the question, include that option only if it is actually present in the user's answer.
- If the user's answer is vague but indicates general interest, produce a broad review/overview topic.
- If the user's answer adds a specific focus, include that focus.
- Do not make the topic more specific than the user's answer allows.
- Keep the proposed query short, clear, and searchable.

Important:
A general answer should not be rewritten as a specific method.

Examples of general answers:
- "geral"
- "visão geral"
- "quero entender a área"
- "quero aprender mais"
- "overview"
- "review"
- "survey"
- "state of the art"

When the answer is general, use expressions such as:
- overview of [topic]
- review of [topic]
- survey on [topic]
- state of the art in [topic]
- visão geral sobre [topic]
- revisão da área de [topic]

Bad behavior:
- Do not turn "visão geral" into a specific method.
- Do not add "voice analysis" unless the user mentioned voice.
- Do not add "facial recognition" unless the user mentioned face or images.
- Do not add "movement detection" unless the user mentioned motion, video, or movement.
- Do not add "deep learning" unless the user mentioned it or it was already in the initial query.
- Do not add a dataset unless the user mentioned it.
- Do not add evaluation metrics unless the user mentioned them.

Good behavior examples:

Initial query: detecção de violência por áudio
Clarification question: Você deseja uma visão geral sobre detecção de violência por áudio ou quer focar em datasets, modelos, métricas ou detecção em tempo real?
Why the question matters: o usuário precisa escolher entre uma visão geral e um foco específico de pesquisa
User answer: visão geral
Output:
{{"proposed_query":"visão geral sobre detecção de violência por áudio","message":"Com base na sua resposta, a busca ficaria: visão geral sobre detecção de violência por áudio","reason":"the user asked for a general overview and the initial query already specifies audio violence detection"}}

Initial query: detecção de violência por áudio
Clarification question: Você deseja uma visão geral sobre detecção de violência por áudio ou quer focar em datasets, modelos, métricas ou detecção em tempo real?
Why the question matters: o usuário precisa escolher entre uma visão geral e um foco específico de pesquisa
User answer: quero entender melhor a área
Output:
{{"proposed_query":"revisão da área de detecção de violência por áudio","message":"Com base na sua resposta, a busca ficaria: revisão da área de detecção de violência por áudio","reason":"the user indicated broad learning intent about the area"}}

Initial query: detecção de violência
Clarification question: Você quer uma visão geral sobre detecção de violência ou quer focar em uma modalidade específica, como áudio, vídeo, texto ou imagens?
Why the question matters: a modalidade muda os termos acadêmicos da busca
User answer: áudio
Output:
{{"proposed_query":"detecção de violência por áudio","message":"Com base na sua resposta, a busca ficaria: detecção de violência por áudio","reason":"the user specified audio as the modality"}}

Initial query: detecção de violência
Clarification question: Você quer uma visão geral sobre detecção de violência ou quer focar em uma modalidade específica, como áudio, vídeo, texto ou imagens?
Why the question matters: a modalidade muda os termos acadêmicos da busca
User answer: visão geral
Output:
{{"proposed_query":"visão geral sobre detecção de violência","message":"Com base na sua resposta, a busca ficaria: visão geral sobre detecção de violência","reason":"the user asked for a general overview without specifying a modality"}}

Initial query: previsão de ações
Clarification question: Você quer uma visão geral sobre previsão de ações ou quer focar em métodos, fontes de dados, horizonte temporal ou métricas?
Why the question matters: diferentes focos levam a diferentes termos acadêmicos de busca
User answer: quero geral, só entender a área
Output:
{{"proposed_query":"visão geral sobre previsão de ações","message":"Com base na sua resposta, a busca ficaria: visão geral sobre previsão de ações","reason":"the user requested a general understanding of the research area"}}

Initial query: previsão de ações
Clarification question: Você quer uma visão geral sobre previsão de ações ou quer focar em métodos, fontes de dados, horizonte temporal ou métricas?
Why the question matters: diferentes focos levam a diferentes termos acadêmicos de busca
User answer: LSTM com notícias e sentimento de mercado
Output:
{{"proposed_query":"previsão de ações usando LSTM, notícias e sentimento de mercado","message":"Com base na sua resposta, a busca ficaria: previsão de ações usando LSTM, notícias e sentimento de mercado","reason":"the user specified the method and data sources"}}

Initial query: dados ruidosos em classificação de imagens médicas
Clarification question: Você quer focar em rótulos ruidosos, ruído na imagem, ruído de aquisição, robustez do modelo ou uma revisão geral do tema?
Why the question matters: diferentes sentidos de ruído levam a diferentes termos acadêmicos de busca
User answer: revisão geral
Output:
{{"proposed_query":"revisão sobre dados ruidosos em classificação de imagens médicas","message":"Com base na sua resposta, a busca ficaria: revisão sobre dados ruidosos em classificação de imagens médicas","reason":"the user asked for a general review of the topic"}}

Initial query: dados ruidosos em classificação de imagens médicas
Clarification question: Você quer focar em rótulos ruidosos, ruído na imagem, ruído de aquisição, robustez do modelo ou uma revisão geral do tema?
Why the question matters: diferentes sentidos de ruído levam a diferentes termos acadêmicos de busca
User answer: rótulos ruidosos
Output:
{{"proposed_query":"rótulos ruidosos em classificação de imagens médicas","message":"Com base na sua resposta, a busca ficaria: rótulos ruidosos em classificação de imagens médicas","reason":"the user clarified that noisy data refers to label noise"}}

Initial query: engenharia de prompt para geração de imagens
Clarification question: Você quer uma visão geral sobre engenharia de prompt para geração de imagens ou quer focar em modelos texto-imagem, qualidade visual, controle de estilo, avaliação de prompts ou técnicas de otimização?
Why the question matters: engenharia de prompt para geração de imagens pode envolver diferentes focos de pesquisa
User answer: quero uma visão geral da área
Output:
{{"proposed_query":"visão geral sobre engenharia de prompt para geração de imagens","message":"Com base na sua resposta, a busca ficaria: visão geral sobre engenharia de prompt para geração de imagens","reason":"the user requested a general overview of the area"}}

Initial query: engenharia de prompt para geração de imagens
Clarification question: Você quer uma visão geral sobre engenharia de prompt para geração de imagens ou quer focar em modelos texto-imagem, qualidade visual, controle de estilo, avaliação de prompts ou técnicas de otimização?
Why the question matters: engenharia de prompt para geração de imagens pode envolver diferentes focos de pesquisa
User answer: quero focar em controle de estilo e qualidade visual
Output:
{{"proposed_query":"engenharia de prompt para controle de estilo e qualidade visual em geração de imagens","message":"Com base na sua resposta, a busca ficaria: engenharia de prompt para controle de estilo e qualidade visual em geração de imagens","reason":"the user specified style control and visual quality as the research focus"}}

Initial query: engenharia de prompt para geração de imagens
Clarification question: Você quer uma visão geral sobre engenharia de prompt para geração de imagens ou quer focar em modelos texto-imagem, qualidade visual, controle de estilo, avaliação de prompts ou técnicas de otimização?
Why the question matters: engenharia de prompt para geração de imagens pode envolver diferentes focos de pesquisa
User answer: avaliação de prompts
Output:
{{"proposed_query":"avaliação de prompts para geração de imagens","message":"Com base na sua resposta, a busca ficaria: avaliação de prompts para geração de imagens","reason":"the user specified prompt evaluation as the focus"}}

Initial query:
{initial_query}

Clarification question:
{question}

Why the question matters:
{question_reason}

User answer:
{user_answer}

Exact shape:
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

                Initial query: {initial_query}
                Previous proposed query: {proposed_query}
                Clarification question: {clarification_question}
                Clarification reason: {clarification_reason}
                Clarification answer: {clarification_answer}
                User revision: {user_revision}

                Exact shape:
                {{"proposed_query":"clear academic search topic","reason":"short reason"}}
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
    """Prompt for initial search queries."""

    text = f"""Return only one JSON object. No markdown. No explanation.

You are planning academic search queries.

User topic:
{context.user_query}

Minimum year:
{context.min_year}

Before writing the queries, analyze internally:
- What is the user's main research goal?
- What area or modality is central to the topic?
- What related academic terms could improve retrieval?
- What meanings should be avoided?
- What would make the search too broad?

Important rules:
- Return only the final JSON object.
- Do not include your internal analysis.
- The "queries" array must contain 1 to 3 non-empty strings.
- Do not simply repeat the user topic unless it is already a strong academic query.
- Every query must preserve the user's main intent.
- If the topic contains a modality, preserve that modality in every query.
- If the topic is about audio, every query must include an audio-related term such as audio, acoustic, sound, or speech.
- Avoid video-only, image-only, and text-only interpretations when the user topic is about audio.
- When the topic is audio violence detection and the user did not explicitly ask for multimodal search, treat it as audio-only.
- For audio-only violence detection, avoid these terms: audio-visual, audiovisual, video, visual, image, multimodal, text, hate speech.
- Prefer precise academic search terms.
- Do not invent datasets, methods, or domains not implied by the topic.
- Keep each query short and searchable.

Good examples:

User topic: audio violence detection
Output:
{{"queries":[
  "audio-only violence detection review",
  "acoustic event detection violence aggression audio-only",
  "sound-based violence detection surveillance review"
]}}

User topic: noisy data in medical image classification
Output:
{{"queries":[
  "noisy labels in medical image classification",
  "robust learning under label noise in medical imaging",
  "review noisy data medical image analysis"
]}}

User topic: medical image classification
Output:
{{"queries":[
  "medical image classification deep learning review",
  "general medical image analysis classification methods",
  "medical imaging datasets for image classification"
]}}

User topic: shortcut bias in deep learning
Output:
{{"queries":[
  "shortcut learning in deep neural networks",
  "spurious correlations and shortcut bias in machine learning",
  "dataset bias robustness shortcut learning review"
]}}

User topic: violence detection
Output:
{{"queries":[
  "violence detection survey",
  "violent event detection in multimedia",
  "deep learning methods for violence detection"
]}}

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


def build_validate_papers_prompt(
    context: SearchContext,
    papers: list[Paper],
    search_feedback: dict[str, object] | None = None,
) -> PromptSpec:
    """Prompt for semantic validation of candidate papers."""

    feedback = search_feedback or {}
    revised_topic = _feedback_text(feedback, "revised_topic") or context.user_query
    positive_constraints = _feedback_list(feedback, "positive_constraints")
    negative_constraints = _feedback_list(feedback, "negative_constraints")
    query_strategy = _feedback_text(feedback, "query_strategy") or "none"
    paper_block = _format_candidate_papers(papers)

    text = f"""Return only one JSON object. No markdown. No explanation.

You are validating academic search results semantically.

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

Candidate papers:
{paper_block}

Task:
For each candidate paper, decide whether it is relevant to the user's search intent.

Rules:
- Evaluate semantic relation to the user's intent, not superficial word overlap.
- Do not mark a paper relevant only because it contains a similar word.
- If the user asked for audio-only, reject video, visual, audio-visual, audiovisual, multimodal, image, text, and hate speech papers unless the title/abstract clearly says audio is the actual violence-detection modality.
- If the user asked for an overview, prioritize surveys, reviews, and papers useful for mapping methods, datasets, or approaches.
- Reject papers from another domain, such as deepfake, ChatGPT, misinformation, hate speech, generic LLMs, data feminism, or text-only detection, when they do not answer the search intent.
- Keep the explanation short and based only on title, abstract, year, source, and URL.
- Do not invent information absent from the title or abstract.
- If the abstract is missing, judge cautiously using only title and metadata.

Relevance labels:
- high: directly related to the user's intent.
- medium: partially related but still useful.
- low: tangential and only worth including if there are few results.
- reject: outside the topic.

Decision labels:
- include: use for high, medium, and only clearly useful low papers.
- exclude: use for reject and papers that violate negative constraints.

Expected behavior for audio-only violence detection:
- Include papers about audio-based, acoustic, sound-based, or audio-signal violence detection.
- Exclude papers about deepfake, ChatGPT, misinformation, Data Feminism, video-based violence detection, vision-based surveillance, and multimodal/audio-visual detection.

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
    """Prompt for refining search queries."""

    evidence = _format_validated_papers(validated_papers[:10])
    used = "; ".join(used_queries[-6:]) or "none"
    feedback = search_feedback or {}
    revised_topic = _feedback_text(feedback, "revised_topic") or context.user_query
    positive_constraints = _feedback_list(feedback, "positive_constraints")
    negative_constraints = _feedback_list(feedback, "negative_constraints")
    query_strategy = _feedback_text(feedback, "query_strategy") or "none"

    text = f"""Return only one JSON object. No markdown. No explanation.
                The "queries" array must contain 1 to 3 non-empty strings.
                Create new academic search queries. Do not repeat used queries.
                Use this revised topic in the query text: {revised_topic}
                Minimum year: {context.min_year}
                Used queries: {used}
                Validated-paper evidence: {evidence}
                Positive constraints: {_format_items(positive_constraints)}
                Negative constraints: {_format_items(negative_constraints)}
                Query strategy: {query_strategy}
                Every query must follow the revised topic, positive constraints, negative constraints, and query strategy.
                Treat negative constraints as hard exclusions.
                When the topic is audio violence detection and the user did not explicitly ask for multimodal search, treat it as audio-only.
                For audio-only violence detection, avoid these terms: audio-visual, audiovisual, video, visual, image, multimodal, text, hate speech.
                Good audio-only query examples:
                - audio-only violence detection review
                - acoustic event detection violence aggression audio-only
                - sound-based violence detection surveillance review
                Exact shape: {{"queries":["{revised_topic} method"],"reason":"new angle"}}
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
- Put desired concepts, modalities, and terms in positive_constraints.
- Put excluded concepts, modalities, and terms in negative_constraints.
- Make query_strategy an actionable sentence for the next query-planning step.
- Keep each constraint short and searchable.
- Preserve the original research intent unless the user explicitly changes it.
- If the user says the search should be audio only, include audio-only/audio-based/acoustic/sound-based terms as positive constraints.
- Use included and excluded papers as evidence for what worked and what failed.
- If the user excludes multimodal or audio-visual directions, include related negative constraints such as multimodal, audio-visual, audiovisual, video, visual, image, text, and hate speech when relevant.

Expected case:
Original user query: audio violence detection
Current refined query: audio violence detection
User feedback: na verdade era para ser sobre audio apenas, sem ser multimodal e sem ser audio-visual
Output:
{{"revised_topic":"audio-only violence detection review","positive_constraints":["audio-only","audio-based","acoustic","sound-based violence detection"],"negative_constraints":["multimodal","audio-visual","audiovisual","video","visual","image","text","hate speech"],"query_strategy":"Search only for papers where violence detection is based on audio signals. Avoid multimedia, visual surveillance, video-based violence detection, and text-based hate speech detection.","reason":"The user clarified that the intended modality is audio only, while the previous ranked papers included multimodal, video, vision-based, and text-related papers."}}

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
