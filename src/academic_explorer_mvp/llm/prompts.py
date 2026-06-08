from dataclasses import dataclass, field

from academic_explorer_mvp.domain.context import SearchContext
from academic_explorer_mvp.domain.paper import RankedPaper


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
REFINE_QUERIES_PROMPT_VERSION = "1.0.1"
CONTINUE_DECISION_PROMPT_VERSION = "1.0.0"
ENRICH_QUERY_PROMPT_VERSION = "1.0.1"
ASSESS_QUERY_CONTEXT_PROMPT_VERSION = "1.0.0"
CONTEXT_QUESTION_PROMPT_VERSION = "1.0.0"
REWRITE_USER_QUERY_PROMPT_VERSION = "1.0.0"
REWRITE_FROM_USER_REVISION_PROMPT_VERSION = "1.0.0"


def build_assess_query_context_prompt(context: SearchContext) -> PromptSpec:
    """Prompt for deciding whether the query has enough search context."""

    text = f"""Return only one JSON object. No markdown. No explanation.

You are preparing an academic paper search.

Task:
Decide whether the user's query has enough context to start building useful academic search queries.

Important:
Do not generate a clarification question here.
Only assess whether the query is clear enough.

Core rule:
A topic being searchable is not enough.

Use has_enough_context=true when:
- the user explicitly asks for a review, survey, overview, introduction, state of the art, or broad learning;
- or the query has a clear technical focus, such as a method, dataset, data source, application, modality, restriction, comparison, metric, or specific research problem.

Use has_enough_context=false when:
- the query is broad and the user did not say whether they want an overview or a specific focus;
- the query could lead to very different academic searches depending on missing context;
- the topic has multiple common meanings and the intended meaning is not clear.

Broad/review intent markers include:
review, survey, overview, literature review, systematic review, introduction,
state of the art, general overview, revisão, revisão geral, visão geral,
panorama, estado da arte, de forma geral, no geral, quero aprender,
quero entender, quero conhecer a área.

Do not include any fields besides has_enough_context and reason.

Examples:

User query: violence detection
Output:
{{"has_enough_context":false,"reason":"the query is broad and does not specify whether the user wants a general review or a specific focus such as audio, video, text, or images"}}

User query: violence detection review
Output:
{{"has_enough_context":true,"reason":"the user explicitly indicated review intent"}}

User query: audio violence detection
Output:
{{"has_enough_context":false,"reason":"the query has a modality, but does not say whether the user wants an overview or a specific focus such as datasets, models, events, or deployment"}}

User query: audio violence detection review
Output:
{{"has_enough_context":true,"reason":"the user explicitly indicated review intent for a clear modality"}}

User query: stock price prediction using LSTM and news sentiment
Output:
{{"has_enough_context":true,"reason":"the query includes task, method, and data source"}}

User query: noisy data in medical image classification
Output:
{{"has_enough_context":false,"reason":"the query is relevant, but noise may refer to label noise, image noise, acquisition noise, or robustness"}}

User query: shortcut bias
Output:
{{"has_enough_context":false,"reason":"the topic is broad and does not indicate overview intent or application domain"}}

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

Core rule:
The question options must be specific to the user's topic.
Do not reuse options from examples unless they make sense for the current topic.

Current user query:
{user_query}

Reason from previous assessment:
{reason}

How to build the question:
1. Identify the real topic of the query.
2. Identify the academic/research area of that topic.
3. Generate options that are natural for that area.
4. Prefer options that would become useful academic search terms.
5. Ask one question only.

Option guidance by topic type:
- For stock prediction, finance, market forecasting, or asset prediction:
  use options such as method, asset type, time horizon, data source, market, evaluation metrics, or review/overview.
  Do not ask about audio, video, image, or text unless the query explicitly mentions multimodal data.

- For violence detection:
  use options such as audio, video, text, images, datasets, models, real-time detection, surveillance, or mobile deployment.

- For medical image classification:
  use options such as image type, disease, organ, dataset, model, label noise, robustness, or evaluation metrics.

- For noisy data:
  use options such as label noise, input noise, acquisition noise, outliers, robust training, or uncertainty.

- For shortcut bias:
  use options such as general overview, computer vision, NLP, medical imaging, dataset bias, spurious correlations, or robustness.

- For recommendation systems:
  use options such as collaborative filtering, content-based filtering, deep learning, cold start, evaluation metrics, or fairness.

- For fake news detection:
  use options such as text-based detection, social network propagation, multimodal detection, datasets, explainability, or language.

Fallback rule:
If the topic does not match any category above, create options from the nouns and technical terms in the user's query.
Prefer:
- method
- application
- dataset
- domain
- metric
- comparison
- time period
- review/overview

Language rule:
Use the same language as the user when possible.
If the query is in Portuguese, ask in Portuguese.
If the query is in English, ask in English.

Question style:
The question should be direct, useful, and specific.

Good patterns:
- "Você quer uma visão geral sobre [topic] ou quer focar em algo mais específico, como [option 1], [option 2], [option 3] ou [option 4]?"
- "Para [topic], você quer priorizar [option 1], [option 2], [option 3] ou uma revisão geral da área?"

Bad questions. Never ask:
- What specific aspect do you need?
- Can you provide more context?
- What do you want to know?
- Please clarify your query.

Also never ask options unrelated to the topic.
For example:
- Do not ask about audio/video/image/text for stock prediction.
- Do not ask about stock assets for medical image classification.
- Do not ask about organs or exams for fake news detection.

Examples:

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

User query: noisy data in medical image classification
Reason from previous assessment: noise may refer to labels, images, acquisition, or robustness
Output:
{{"question":"Em dados ruidosos para classificação de imagens médicas, você quer focar em rótulos ruidosos, ruído na imagem, ruído de aquisição, robustez do modelo ou uma revisão geral do tema?","reason":"different meanings of noise lead to different academic search terms"}}

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


def build_refine_queries_prompt(
    context: SearchContext,
    ranked_papers: list[RankedPaper],
    used_queries: list[str],
    paper_feedback: str = "",
) -> PromptSpec:
    """Prompt for refining search queries."""

    titles = "; ".join(item.paper.title[:80] for item in ranked_papers) or "none"
    used = "; ".join(used_queries[-6:]) or "none"
    feedback = paper_feedback or "none"

    text = f"""Return only one JSON object. No markdown. No explanation.
                The "queries" array must contain 1 to 3 non-empty strings.
                Create new academic search queries. Do not repeat used queries.
                Use this topic in the query text: {context.user_query}
                Minimum year: {context.min_year}
                Used queries: {used}
                Best paper titles: {titles}
                User feedback/restriction for the next search: {feedback}
                If user feedback/restriction is not "none", every query must obey it.
                Treat exclusions in the feedback as hard exclusions.
                When the topic is audio violence detection and the user did not explicitly ask for multimodal search, treat it as audio-only.
                For audio-only violence detection, avoid these terms: audio-visual, audiovisual, video, visual, image, multimodal, text, hate speech.
                Good audio-only query examples:
                - audio-only violence detection review
                - acoustic event detection violence aggression audio-only
                - sound-based violence detection surveillance review
                Exact shape: {{"queries":["{context.user_query} method"],"reason":"new angle"}}
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


def build_continue_decision_prompt(
    context: SearchContext,
    round_number: int,
    ranked_papers: list[RankedPaper],
    last_new_paper_count: int,
    last_new_useful_count: int,
) -> PromptSpec:
    """Prompt for deciding whether another search round is useful."""

    titles = "; ".join(item.paper.title[:80] for item in ranked_papers) or "none"

    text = f"""Return only one JSON object. No markdown. No explanation.
                Decide if another academic search round is useful.
                Topic: {context.user_query}
                Round: {round_number} of {context.max_rounds}
                New papers: {last_new_paper_count}
                New useful papers: {last_new_useful_count}
                Best paper titles: {titles}
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
