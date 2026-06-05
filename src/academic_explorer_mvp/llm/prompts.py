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


INITIAL_QUERIES_PROMPT_VERSION = "1.0.2"
REFINE_QUERIES_PROMPT_VERSION = "1.0.0"
CONTINUE_DECISION_PROMPT_VERSION = "1.0.0"
ENRICH_QUERY_PROMPT_VERSION = "1.0.1"
REWRITE_USER_QUERY_PROMPT_VERSION = "1.0.0"
REWRITE_FROM_USER_REVISION_PROMPT_VERSION = "1.0.0"

def build_enrich_query_prompt(context: SearchContext) -> PromptSpec:
    """Prompt for assessing whether the user's topic needs clarification."""

    text = f"""Return only one JSON object. No markdown. No explanation.

You are preparing an academic paper search.

Your task is to decide whether the user's intent is clear enough to start the search,
or whether one clarification question should be asked first.

Core rule:
A topic being searchable is not enough.

can_search=true means:
The user's search intention is clear enough to generate useful academic search queries.

can_search=false means:
The query is broad or ambiguous, and one clarification question would help define the search direction.

Decision process:
1. Identify the main topic.
2. Decide whether the topic is broad or specific.
3. If the topic is broad, check whether the user explicitly wants a broad search.
4. If the topic is broad and the user did not explicitly say they want a broad search, use can_search=false.
5. If the topic is broad but the user clearly wants overview, review, survey, general understanding, or learning the field, use can_search=true.
6. If the topic includes a clear method, domain, modality, data type, application, restriction, or research focus, use can_search=true.

Important:
Do not infer that the user wants a broad review just because the query is broad.
The user must explicitly indicate broad intent.

Explicit broad intent can appear through meanings like:
- review
- survey
- overview
- general
- broad
- introduction
- tutorial
- landscape
- state of the art
- literature review
- systematic review
- visão geral
- revisão
- panorama
- estado da arte
- de forma geral
- no geral
- introdução
- quero aprender sobre
- quero entender
- quero conhecer a área
- entender o mundo de
- entender os nichos
- explorar a área

Broad queries usually need clarification when there is no explicit broad intent.
Examples of broad queries:
- stock prediction
- violence detection
- medical image classification
- noisy data
- shortcut bias
- sentiment analysis
- fraud detection
- recommendation systems
- fake news detection
- disease prediction
- emotion recognition

When asking a clarification question:
- ask only one question;
- do not ask a generic question;
- include useful options;
- help the user choose between broad overview and a specific focus;
- use the same language as the user when possible;
- the question must be directly related to the current user query;
- the question must mention the current topic or terms from the current topic;
- the question must help discover better academic search terms.

Good clarification question pattern:
"Você quer uma visão geral/revisão sobre [current topic] ou quer focar em algo mais específico, como [option 1], [option 2], [option 3] ou [option 4]?"

Bad questions. Never ask:
- What specific aspect do you need?
- Can you provide more context?
- What do you want to know?
- Please clarify your query.

Examples:

User query: stock prediction
Output:
{{"can_search":false,"question":"Você quer uma visão geral/revisão sobre previsão de ações ou quer focar em algo mais específico, como método, tipo de ativo, horizonte temporal, fonte de dados ou métricas?","reason":"the query is broad and does not explicitly say whether the user wants an overview or a specific focus"}}

User query: stock prediction review
Output:
{{"can_search":true,"question":null,"reason":"the user explicitly indicated a review intent"}}

User query: quero aprender sobre o mundo de predição de ações de forma geral
Output:
{{"can_search":true,"question":null,"reason":"the user explicitly indicated broad learning intent"}}

User query: stock price prediction using LSTM and news sentiment
Output:
{{"can_search":true,"question":null,"reason":"the query includes task, method, and data source"}}

User query: violence detection
Output:
{{"can_search":false,"question":"Você quer uma visão geral sobre detecção de violência ou quer focar em uma modalidade específica, como áudio, vídeo, texto ou imagens?","reason":"the query is broad and the modality or overview intent would strongly change the search terms"}}

User query: audio violence detection
Output:
{{"can_search":false,"question":"Você quer uma visão geral sobre detecção de violência em áudio ou quer focar em algo mais específico, como gritos, brigas, fala/emoção, datasets, modelos ou aplicação em celular?","reason":"the query has modality and task, but does not explicitly say whether the user wants overview or a specific technical focus"}}

User query: audio violence detection review
Output:
{{"can_search":true,"question":null,"reason":"the user explicitly indicated review intent for a clear modality and task"}}

User query: medical image classification
Output:
{{"can_search":false,"question":"Você quer uma visão geral sobre classificação de imagens médicas ou quer focar em algum tipo de imagem, como raio-X, ressonância, tomografia, histopatologia ou retina?","reason":"the query is broad and does not indicate overview intent or image type"}}

User query: general medical image classification review
Output:
{{"can_search":true,"question":null,"reason":"the user explicitly indicated general review intent"}}

User query: noisy data in medical image classification
Output:
{{"can_search":false,"question":"Você quer uma visão geral ou quer focar em rótulos ruidosos, imagens com ruído, robustez de modelos ou algum tipo específico de imagem médica?","reason":"the query has domain and problem, but different meanings of noise lead to different search terms"}}

User query: shortcut bias
Output:
{{"can_search":false,"question":"Você quer uma visão geral sobre shortcut bias ou quer focar em algum domínio, como imagens médicas, NLP ou visão computacional?","reason":"the query is broad and does not indicate overview intent or domain"}}

User query: quero uma visão geral sobre shortcut bias
Output:
{{"can_search":true,"question":null,"reason":"the user explicitly indicated overview intent"}}

The examples above are only examples.
Do not copy an example unless the current user query has the same topic.
Your answer must be about the current user query below.
The clarification question must mention the current topic or a direct translation of it.
If the current query is about stock prediction, ask about stock prediction.
If the current query is about violence detection, ask about violence detection.
If the current query is about medical images, ask about medical images.
Never answer about shortcut bias, medical images, violence detection, or any other example topic unless that is the current user query.

Current user query:
{context.user_query}

Final check before answering:
- If the query is broad and has no explicit broad-intent marker, return can_search=false.
- If can_search=false, write a useful clarification question with options related to the current user query.
- If the user clearly asks for general understanding, overview, review, survey, or learning the field, return can_search=true.
- If can_search=true, question must be null.
- If can_search=false, question must be a real question and must not be null.
- Return only one valid JSON object.

Required JSON shape:
{{"can_search":false,"question":"real clarification question about the current user query with options?","reason":"short reason"}}

or:
{{"can_search":true,"question":null,"reason":"short reason"}}
"""

    return PromptSpec(
        name="enrich_query",
        version=ENRICH_QUERY_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "initial_query_assessment",
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
                Create one clearer academic search topic from the initial query and
                the user's clarification answer.

                This output is not a list of academic search queries. It is the
                refined user topic that will later be passed to initial query
                planning.

                Rules:
                - Create only one clearer academic topic.
                - Do not create multiple queries.
                - Do not invent information.
                - Use only the initial query and the user's answer.

                Initial query: {initial_query}
                Clarification question: {question}
                Why the question matters: {question_reason}
                User answer: {user_answer}

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
- Prefer precise academic search terms.
- Do not invent datasets, methods, or domains not implied by the topic.
- Keep each query short and searchable.

Good examples:

User topic: audio violence detection
Output:
{{"queries":[
  "audio violence detection",
  "acoustic event detection for violence and aggression",
  "audio surveillance scream gunshot violence detection"
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
) -> PromptSpec:
    """Prompt for refining search queries."""

    titles = "; ".join(item.paper.title[:80] for item in ranked_papers) or "none"
    used = "; ".join(used_queries[-6:]) or "none"

    text = f"""Return only one JSON object. No markdown. No explanation.
                The "queries" array must contain 1 to 3 non-empty strings.
                Create new academic search queries. Do not repeat used queries.
                Use this topic in the query text: {context.user_query}
                Minimum year: {context.min_year}
                Used queries: {used}
                Best paper titles: {titles}
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
