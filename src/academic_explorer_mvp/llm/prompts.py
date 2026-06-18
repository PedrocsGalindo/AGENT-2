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
    """Prompt for deciding whether the query has enough search context."""

    text = f"""Return only one JSON object. No markdown. No explanation.

You are preparing an academic paper search.

Task:
Decide whether the user's query has enough context to start building useful academic search queries.

Language rule:
- The reason must be in English because it is internal.
- Keep JSON keys in English.

Core decision criterion:
A query has enough context only when it contains:
1. a recognizable academic topic; and
2. enough search intent to avoid guessing the user's desired direction.

Searchable is not the same as enough context.
If a general search can be created, but the desired meaning or direction is ambiguous, context is missing.

Use has_enough_context=true when:
- the user explicitly asks for a general search, overview, introduction, broad learning, trend analysis, review, survey, or state of the art;
- or the query contains a clear topic plus a specific research focus enough to guide the search, such as method, dataset, data source, metric, comparison, restriction, application, domain, or specific problem. A modality alone is not enough;
- and academic search queries can be generated without guessing the intended meaning.

Use has_enough_context=false when:
- the query is broad and the user did not say whether they want a general direction or a specific focus;
- the query contains an ambiguous term that can lead to substantially different academic searches;
- the topic could naturally mean different methods, domains, modalities, systems, data sources, or applications;
- generating queries would require assuming the user's intended meaning.

Ambiguity rule:
If a key term is ambiguous, return has_enough_context=false and state the likely ambiguity in the reason.
Examples of ambiguous terms include terms such as "agent", "bias", "automation", "shopping automation", "noise", "detection", "security", or "prediction" when the intended meaning is not clear.

General intent rule:
If the query explicitly says "general", "overview", "visão geral", "geral", "quero entender", "trend", "trends", "state of the art", "survey", or "review", this is enough context.
Do not force review/survey as the only possible output later; just recognize that the user wants a broad search.

Examples:

User query: stock prediction
Output:
{{"has_enough_context":false,"reason":"the query is searchable, but it does not specify whether the user wants a general search or a specific focus such as methods, data sources, assets, markets, time horizon, or metrics"}}

User query: stock prediction geral
Output:
{{"has_enough_context":true,"reason":"the user explicitly indicated a broad general direction for the topic"}}

User query: stock price prediction using LSTM and news sentiment
Output:
{{"has_enough_context":true,"reason":"the query specifies the task, method, and data source"}}

User query: automated shopping
Output:
{{"has_enough_context":false,"reason":"the term automated shopping is ambiguous and may refer to shopping agents, e-commerce automation, checkout automation, recommendation systems, or retail operations"}}

User query: shopping agents overview
Output:
{{"has_enough_context":true,"reason":"the query specifies shopping agents and indicates broad overview intent"}}

User query: audio violence detection
Output:
{{"has_enough_context":false,"reason":"the query specifies the task and modality, but not whether the user wants a general direction or a specific focus such as models, datasets, features, or deployment"}}

User query: audio violence detection geral
Output:
{{"has_enough_context":true,"reason":"the query specifies audio violence detection and indicates broad general intent"}}

User query: noisy data in medical image classification
Output:
{{"has_enough_context":false,"reason":"the query is relevant, but noisy data may refer to label noise, image noise, acquisition noise, outliers, or robustness"}}

User query: label noise in medical image classification
Output:
{{"has_enough_context":true,"reason":"the query specifies the type of noise and the application domain"}}

User query:
{context.user_query}

Required JSON shape:
{{"has_enough_context":true,"reason":"short reason"}}
or
{{"has_enough_context":false,"reason":"short reason"}}
"""

    return PromptSpec(
        name="assess_query_context",
        version=ASSESS_QUERY_CONTEXT_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "query_context_assessment"},
    )


def build_context_question_prompt(user_query: str, reason: str) -> PromptSpec:
    """Prompt for generating one clarification question with domain-specific options."""

    text = f"""Return only one JSON object. No markdown. No explanation.

You are preparing an academic paper search.

Task:
Generate exactly one useful clarification question for the user's current query.

Important:
- Do not reassess whether the query has enough context.
- The assessment was already done.
- Ask one question only.
- The question must help identify the user's intended academic search direction.

Current user query:
{user_query}

Reason from previous assessment:
{reason}

Language rule:
- The question must be in Portuguese because it is shown to the user.
- The reason must be in English because it is internal.
- Keep JSON keys in English.

Core rules:
- Build the question from the actual user query and the missing context in the reason.
- If the reason indicates ambiguity, explicitly mention the ambiguity and offer likely meanings.
- If the reason indicates broadness, ask whether the user wants a general direction or a specific focus.
- Do not copy examples literally.
- Do not ask generic questions such as "what do you want to know?".
- Do not add unrelated options.
- Do not ask about a modality if the modality is already clear; ask about focus instead.

Useful dimensions when relevant:
- general direction, method, model family, application, domain, modality, dataset, data source, metric, comparison, population, disease, task, system type, deployment context, evaluation focus, theoretical vs practical focus.

Good question patterns:
- "O termo [term] pode significar [meaning 1], [meaning 2] ou [meaning 3]. Qual sentido você quer priorizar?"
- "Você quer uma visão geral sobre [topic] ou quer focar em algo mais específico, como [option 1], [option 2], [option 3] ou [option 4]?"
- "Em [topic], o foco deve ser [option 1], [option 2], [option 3], [option 4] ou uma visão geral?"

Bad questions. Never ask:
- What specific aspect do you need?
- Can you provide more context?
- What do you want to know?
- Please clarify your query.
- Could you be more specific?

Topic guidance:
- For prediction/forecasting: methods, data sources, target variable, time horizon, market/domain, metrics, or general direction.
- For detection/classification: modality, data type, datasets, models, feature extraction, real-time deployment, benchmarks, metrics, or general direction.
- For ambiguous technical terms: list the likely meanings and ask the user to choose.
- For agents: distinguish LLM/VLM agents, software agents, autonomous systems, recommendation agents, shopping agents, robotics, or workflows when relevant.
- For bias: distinguish statistical bias, dataset bias, shortcut bias, fairness bias, model bias, medical bias, social bias, or evaluation bias when relevant.
- For noise: distinguish label noise, input noise, acquisition noise, outliers, noisy environments, or robustness.

Examples:

User query: automated shopping
Reason: the term automated shopping is ambiguous and may refer to shopping agents, e-commerce automation, checkout automation, recommendation systems, or retail operations
Output:
{{"question":"O termo \"automated shopping\" ficou ambíguo: você quer falar de agentes de compra com IA, automação em e-commerce, checkout automático, sistemas de recomendação ou operações de varejo?","reason":"the question exposes the ambiguity and offers likely academic interpretations"}}

User query: audio violence detection
Reason: the query specifies the task and modality, but not whether the user wants a general direction or a specific focus
Output:
{{"question":"Você quer uma visão geral sobre detecção de violência por áudio ou quer focar em algo mais específico, como modelos, datasets, extração de características, métricas ou detecção em tempo real?","reason":"the question preserves the audio modality and asks about the missing research focus"}}

User query: noisy data in medical image classification
Reason: noisy data may refer to labels, images, acquisition, outliers, or robustness
Output:
{{"question":"Em dados ruidosos para classificação de imagens médicas, você quer focar em rótulos ruidosos, ruído na imagem, ruído de aquisição, outliers, robustez do modelo ou uma visão geral?","reason":"the question asks about the specific meaning of noisy data in the user's topic"}}

Now generate the final answer for the current user query only.

Required JSON shape:
{{"question":"one clarification question with topic-specific options","reason":"short reason"}}
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
    """Prompt for proposing one refined user topic after clarification."""

    text = f"""Return only one JSON object. No markdown. No explanation.

You are preparing an academic paper search.

Task:
Create one clearer academic search topic from the initial query and the user's clarification answer.

This output is not a list of search queries.
It is one refined user topic that will later be passed to query planning and semantic validation.

Input language:
The initial query, clarification question, and user answer may be in Portuguese, English, or mixed language.

Output language:
- proposed_query must be in English because it will be used for academic paper search.
- message must be in Portuguese because it is shown to the user.
- reason must be in English because it is internal.
- JSON keys must stay in English.

Main goal:
Combine the initial query with the user's clarification answer.
Use only information actually provided by the user.
Do not invent methods, datasets, metrics, domains, applications, modalities, or restrictions.

Core rules:
- Create exactly one refined academic topic.
- Preserve the main topic from the initial query.
- Preserve useful technical terms from the initial query.
- Use the clarification question only to understand what the user's answer refers to.
- Translate the intended meaning into English for proposed_query.
- The proposed_query should be direct and faithful.
- The proposed_query does not need to be very short, but it must not become verbose.
- Include as much user-provided context as useful for academic search.
- Do not create multiple search queries.
- Do not include inferred restrictions as separate fields.
- Do not make the topic more specific than the user's answer allows.
- If the user's answer adds a specific focus, include that focus.
- If the user's answer is broad or general, keep the refined topic broad.
- If the user selects one option from the question, include only that selected option.
- If the user rejects an option, do not include it.

General answer rule:
If the user says something equivalent to "geral", "visão geral", "quero entender", "panorama", "algo amplo", "não sei ainda", or "general":
- keep the topic broad;
- do not automatically turn the refined topic into review, survey, or systematic review;
- use terms such as "general" or "broad" only when they help preserve the user's meaning;
- do not add a specific method, dataset, metric, or restriction.

Review/survey distinction:
Only use "review", "survey", "systematic review", or "state of the art" in proposed_query if the user explicitly asks for review, survey, systematic review, literature review, or state of the art.
A general answer is not the same as asking only for review papers.

Negative rules:
- Do not add "deep learning" unless the user mentioned it or the topic clearly requires it.
- Do not add "machine learning" unless the user mentioned it or the topic clearly involves computational modeling.
- Do not add a dataset unless the user mentioned it.
- Do not add evaluation metrics unless the user mentioned evaluation or metrics.
- Do not add a modality unless it appears in the initial query, clarification question, or user answer.
- Do not add an application domain unless it appears in the initial query or user answer.
- Do not transform a broad answer into a narrow method.
- Do not transform a domain answer into a method answer.

Examples:

Initial query: audio violence detection
Clarification question: Você quer uma visão geral sobre detecção de violência por áudio ou quer focar em modelos, datasets, extração de características, métricas ou detecção em tempo real?
User answer: geral
Output:
{{"proposed_query":"general audio-based violence detection","message":"Com base na sua resposta, a busca ficaria: general audio-based violence detection","reason":"the user asked for a broad direction while preserving the audio-based violence detection topic"}}

Initial query: audio violence detection
Clarification question: Você quer uma visão geral sobre detecção de violência por áudio ou quer focar em modelos, datasets, extração de características, métricas ou detecção em tempo real?
User answer: revisão sistemática
Output:
{{"proposed_query":"systematic review of audio-based violence detection","message":"Com base na sua resposta, a busca ficaria: systematic review of audio-based violence detection","reason":"the user explicitly asked for a systematic review"}}

Initial query: automated shopping
Clarification question: O termo "automated shopping" ficou ambíguo: você quer falar de agentes de compra com IA, automação em e-commerce, checkout automático, sistemas de recomendação ou operações de varejo?
User answer: agentes de compra com LLM e VLM
Output:
{{"proposed_query":"shopping agents using LLMs and VLMs","message":"Com base na sua resposta, a busca ficaria: shopping agents using LLMs and VLMs","reason":"the user clarified that automated shopping means AI shopping agents using LLMs and VLMs"}}

Initial query: shortcut bias
Clarification question: Você quer uma visão geral sobre shortcut bias ou quer focar em visão computacional, NLP, imagens médicas, viés de dataset, correlações espúrias ou robustez?
User answer: imagens médicas
Output:
{{"proposed_query":"shortcut bias in medical imaging","message":"Com base na sua resposta, a busca ficaria: shortcut bias in medical imaging","reason":"the user selected medical imaging as the application context"}}

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
    """Prompt for refining a user-written revision of the proposed topic."""

    text = f"""Return only one JSON object. No markdown. No explanation.

Create one clear academic search topic from the user's revision.

Rules:
- Prefer the revision written by the user.
- Preserve useful context from the initial query and clarification only when it is compatible with the revision.
- Do not invent methods, datasets, domains, restrictions, or modalities.
- Do not create multiple queries.
- The proposed_query must be in English because it will be used for academic paper search.
- The message must be in Portuguese because it is shown to the user.
- The reason must be in English because it is internal.
- Keep JSON keys in English.
- The proposed_query should be direct, faithful, and not verbose.

Initial query: {initial_query}
Previous proposed query: {proposed_query}
Clarification question: {clarification_question}
Clarification reason: {clarification_reason}
Clarification answer: {clarification_answer}
User revision: {user_revision}

Required JSON shape:
{{"proposed_query":"clear academic search topic","message":"Com base na sua revisão, a busca ficaria: ...","reason":"short reason"}}
"""

    return PromptSpec(
        name="rewrite_from_user_revision",
        version=REWRITE_FROM_USER_REVISION_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "user_revision_query_rewrite"},
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
Create exactly 3 academic search queries for finding candidate papers.

The user topic was already assessed or refined before reaching this step.
Do not ask questions here.
Do not reassess whether the topic has enough context.

Search philosophy:
Favor recall over precision because a later semantic validation step will filter irrelevant papers.
The search should be broad enough to discover useful papers, but still faithful to the user's topic.

Core goal:
Generate 3 meaningfully different search queries.
Do not produce three queries that only swap one word.
Do not drift into unrelated modalities, methods, domains, or applications.

Internal planning steps:
1. Identify the main research intent.
2. Identify core anchors: terms or concepts that must remain in every query, using exact terms or close synonyms.
3. Identify optional technical terms: useful terms that may appear in some queries but not all.
4. Identify likely academic synonyms.
5. Identify terms that would cause topic drift.

Core anchors:
- Core anchors must appear in every query, either exactly or as close academic synonyms.
- A core anchor is usually the main task, object, modality, domain, or problem without which the query stops representing the user topic.
- Examples:
  - audio violence detection: core anchors are violence detection and audio/sound/acoustic.
  - bias in medical image classification: core anchors are bias and medical image classification/medical imaging classification.
  - shopping agents: core anchors are shopping/automated shopping and agents/assistants.

Optional technical terms:
- Optional technical terms should appear in some queries, not necessarily all.
- Examples: LLM, VLM, RAG, LSTM, transformer, benchmark, dataset, state of the art, trend, survey.
- Use optional terms according to their importance in the topic.

General-topic behavior:
If the user topic indicates "general", "geral", "visão geral", "overview", "broad", "panorama", or similar:
- keep the search broad;
- exactly one query may include a term such as "state of the art", "trends", "overview", "survey", or "review" when useful;
- do not put review/survey/state-of-the-art terms in all queries unless the user explicitly asked only for reviews or surveys.

Query structure:
- Query 1: closest faithful academic query.
- Query 2: synonym or equivalent academic wording.
- Query 3: broader exploratory angle, such as trends, state of the art, systems, applications, or related terminology, only when faithful.

Rules:
- Return only the final JSON object.
- The queries array must contain exactly 3 non-empty strings.
- Academic search queries must be in English.
- The reason must be in English because it is internal.
- Every query must preserve the user's main intent.
- Keep queries direct and searchable.
- Do not invent datasets, metrics, years, restrictions, or narrow applications.
- Do not add a specific method unless the user mentioned it or it is central to the topic.
- Do not add a modality unless the user mentioned it or it is part of the topic.
- Do not broaden a specific topic into an unrelated generic field.
- Do not narrow a broad topic into one specific method.
- Do not copy examples unless they fit the current topic.

Examples:

User topic: general audio-based violence detection
Output:
{{"queries":[
  "audio-based violence detection",
  "acoustic violent event detection",
  "state of the art sound-based violence detection"
],"reason":"the queries preserve violence detection and audio-related anchors while using one broader state-of-the-art query for general exploration"}}

User topic: shopping agents using LLMs and VLMs
Output:
{{"queries":[
  "AI shopping agents",
  "LLM agents for online shopping",
  "VLM shopping assistants in e-commerce"
],"reason":"the queries preserve shopping agents while using LLM and VLM as important but not universal optional terms"}}

User topic: bias in medical image classification
Output:
{{"queries":[
  "bias in medical image classification",
  "dataset bias in medical imaging classification",
  "shortcut learning in medical image classification"
],"reason":"the queries preserve the bias and medical image classification anchors while exploring related bias terminology"}}

User topic: stock price prediction using LSTM and news sentiment
Output:
{{"queries":[
  "stock price prediction using LSTM and news sentiment",
  "stock market forecasting with recurrent neural networks and sentiment analysis",
  "financial time series prediction using news sentiment"
],"reason":"the queries preserve the stock prediction task while varying method and data-source wording"}}

User topic: label noise in medical image classification
Output:
{{"queries":[
  "label noise in medical image classification",
  "medical image classification with noisy labels",
  "robust learning under label noise in medical imaging"
],"reason":"the queries preserve label noise and medical image classification while exploring robust-learning terminology"}}

Required JSON shape:
{{"queries":["academic search query 1","academic search query 2","academic search query 3"],"reason":"short reason"}}
"""

    return PromptSpec(
        name="initial_queries",
        version=INITIAL_QUERIES_PROMPT_VERSION,
        text=text,
        metadata={"output_format": "json", "purpose": "initial_query_planning"},
    )


def build_plan_filters_prompt(context: SearchContext) -> PromptSpec:
    """Prompt for planning semantic validation filters."""

    text = f"""Return only one valid JSON object. No markdown. No explanation.

You are planning semantic filters for academic paper validation.

Refined user topic:
{context.user_query}

Task:
Create compact semantic validation filters from the refined user topic.

These filters will later be used to decide whether candidate papers match the user's academic intent.

Important:
- Do not generate search queries.
- Do not validate papers here.
- Do not invent a new topic.
- Do not rename JSON keys.
- Always include all required keys.
- All fields except primary_intent and reason must be lists.
- If a list field has no values, return an empty list.
- Do not add extra keys.

Language rule:
- All JSON values must be in English.
- JSON keys must be in English.

Official JSON schema:
{{
  "primary_intent": "string",
  "conservative_filters": ["string"],
  "expansive_filters": ["string"],
  "negative_constraints": ["string"],
  "not_inferred": ["string"],
  "validation_priority": ["string"],
  "reason": "string"
}}

Field meaning:
- primary_intent: the central academic intent of the user topic.
- conservative_filters: central criteria that a relevant paper should satisfy.
- expansive_filters: related concepts that may help identify useful papers, but are not mandatory by themselves.
- negative_constraints: concepts, domains, tasks, or interpretations that indicate mismatch.
- not_inferred: things the model must not assume because the user did not provide them.
- validation_priority: ordered criteria for validating candidate papers.
- reason: one short reason explaining the filters.

Core filtering principle:
A paper must first satisfy the conservative_filters.
Expansive filters are only supporting evidence.
Do not include a paper only because it matches an expansive filter.
Do not accept a paper only because it shares one isolated word with the topic.

Conservative filter rules:
- Use only the core task, object, system, modality, domain, or method that is explicit or strongly implied.
- Keep conservative_filters narrow and central.
- Do not put generic terms such as "AI", "machine learning", "automation", "market analysis", "customer experience", or "data analytics" as conservative filters unless they are truly central to the user's topic.
- If a term is broad, pair it with the user's actual object or task.
  Example: use "shopping agent", not just "agent".
  Example: use "automated shopping", not just "automation".
  Example: use "audio-based violence detection", not just "audio".

Expansive filter rules:
- Use expansive_filters for related concepts that can help recall.
- Expansive filters must still be close to the user's topic.
- Do not add business, marketing, logistics, or social concepts unless the user asked for them.
- Do not add specific implementation details unless they are natural and useful for the topic.

Negative constraint rules:
- Add likely confusions caused by ambiguous words.
- Add nearby topics that would produce irrelevant papers.
- Add domains that share terms but do not match the user's intent.
- If the topic contains "agent", exclude unrelated uses such as principal-agent economics, generic agent-based modeling, central banks as agents, or social agents unless the user asks for them.
- If the topic is about shopping agents, exclude warehouse automation, supply chain, inventory optimization, generic customer experience, purchase intention, brand engagement, advertising, and retail strategy when they do not involve a shopping/purchasing agent.

Not inferred rules:
Use not_inferred for tempting but unsupported assumptions.
Examples:
- specific dataset
- specific model architecture
- specific platform
- user study
- web scraping
- API integration
- real-time deployment
- mobile app
- recommender system
Only put something in not_inferred if it would be tempting to assume but was not stated.

Computer Science relevance:
If the topic is computational, prefer papers with clear computational contributions, such as algorithms, models, agents, systems, datasets, benchmarks, experiments, architectures, pipelines, software, automation workflows, or technical evaluation.
Do not force "machine learning" or "deep learning" unless the user topic supports it.

Validation logic:
The validation_priority should tell the validator what to check first.
The first priority must be the core user intent.
Later priorities should check supporting concepts and exclusions.

Length limits:
- conservative_filters: max 4 items.
- expansive_filters: max 8 items.
- negative_constraints: max 10 items.
- not_inferred: max 8 items.
- validation_priority: max 5 items.
- reason: one short sentence.

Examples:

Topic: automatic shopping agent
Output:
{{
  "primary_intent": "academic search about automatic shopping agents",
  "conservative_filters": [
    "shopping agent",
    "automated shopping",
    "product search or purchase assistance"
  ],
  "expansive_filters": [
    "AI shopping assistant",
    "autonomous purchasing assistant",
    "product comparison",
    "price comparison",
    "deal finding",
    "web-based product search",
    "LLM agent for shopping",
    "browser agent for online shopping"
  ],
  "negative_constraints": [
    "inventory optimization",
    "warehouse automation",
    "supply chain management",
    "order fulfillment without shopping agent",
    "customer experience without shopping agent",
    "purchase intention without automated agent",
    "brand engagement",
    "advertising strategy",
    "principal-agent economics",
    "generic autonomous agents without shopping context"
  ],
  "not_inferred": [
    "specific platform",
    "specific dataset",
    "specific model architecture",
    "web scraping",
    "API integration",
    "VLM agent",
    "recommendation system",
    "checkout automation"
  ],
  "validation_priority": [
    "match shopping agent or automated shopping intent",
    "check product search or purchase assistance role",
    "check computational agent or system contribution",
    "apply negative constraints",
    "use expansive filters only as supporting evidence"
  ],
  "reason": "The topic is about an automated agent that helps with shopping or purchasing, not general retail analytics or logistics."
}}

Topic: shopping agents using LLMs and VLMs
Output:
{{
  "primary_intent": "academic search about shopping agents using LLMs or VLMs",
  "conservative_filters": [
    "shopping agent",
    "LLM or VLM agent",
    "online shopping or purchasing assistance"
  ],
  "expansive_filters": [
    "AI shopping assistant",
    "multimodal shopping assistant",
    "product search agent",
    "product comparison",
    "price comparison",
    "web automation for shopping",
    "browser agent",
    "recommendation agent"
  ],
  "negative_constraints": [
    "generic LLM agents without shopping context",
    "generic VLM papers without shopping context",
    "warehouse automation",
    "inventory optimization",
    "supply chain management",
    "customer experience without agent system",
    "virtual reality shopping without agents",
    "principal-agent economics"
  ],
  "not_inferred": [
    "specific dataset",
    "specific benchmark",
    "specific e-commerce platform",
    "web scraping",
    "API integration",
    "checkout automation"
  ],
  "validation_priority": [
    "match shopping agent intent",
    "match LLM or VLM agent aspect",
    "check shopping or purchasing context",
    "check computational contribution",
    "apply negative constraints"
  ],
  "reason": "The topic requires both shopping-agent context and LLM/VLM-based agent technology."
}}

Topic: general audio-based violence detection
Output:
{{
  "primary_intent": "general search about audio-based violence detection",
  "conservative_filters": [
    "violence detection",
    "audio-based detection"
  ],
  "expansive_filters": [
    "sound-based violent event detection",
    "acoustic event detection",
    "audio signal processing",
    "audio classification",
    "aggression detection"
  ],
  "negative_constraints": [
    "video-only violence detection",
    "image-only violence detection",
    "visual surveillance",
    "audio-visual violence detection",
    "multimodal violence detection",
    "hate speech detection",
    "deepfake detection",
    "non-computational violence studies"
  ],
  "not_inferred": [
    "specific dataset",
    "specific model architecture",
    "real-time deployment",
    "multimodal learning"
  ],
  "validation_priority": [
    "match violence detection task",
    "match audio modality",
    "check computational contribution",
    "apply negative constraints",
    "use expansive filters only as supporting evidence"
  ],
  "reason": "The user wants a general search about violence detection based on audio, without assuming a specific method or dataset."
}}

Topic: bias in medical image classification
Output:
{{
  "primary_intent": "general search about bias in medical image classification",
  "conservative_filters": [
    "bias",
    "medical image classification"
  ],
  "expansive_filters": [
    "dataset bias",
    "shortcut learning",
    "spurious correlations",
    "medical imaging",
    "robustness",
    "model generalization"
  ],
  "negative_constraints": [
    "natural image classification without medical context",
    "medical imaging without classification",
    "fairness discussion without computational evaluation",
    "clinical bias without machine learning or classification"
  ],
  "not_inferred": [
    "specific disease",
    "specific imaging modality",
    "specific model architecture",
    "specific dataset"
  ],
  "validation_priority": [
    "match bias-related problem",
    "match medical image classification",
    "check computational contribution",
    "apply negative constraints"
  ],
  "reason": "The topic requires bias and medical image classification, while related robustness concepts are useful but not mandatory."
}}

Now return the JSON object for the current refined user topic.

Required JSON shape:
{{
  "primary_intent": "string",
  "conservative_filters": ["string"],
  "expansive_filters": ["string"],
  "negative_constraints": ["string"],
  "not_inferred": ["string"],
  "validation_priority": ["string"],
  "reason": "string"
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
    feedback_negative_constraints = _feedback_list(feedback, "negative_constraints")
    query_strategy = _feedback_text(feedback, "query_strategy") or "none"

    primary_intent = _feedback_text(filters, "primary_intent") or "none"
    conservative_filters = _feedback_list(filters, "conservative_filters")
    expansive_filters = _feedback_list(filters, "expansive_filters")
    negative_constraints = [
        *_feedback_list(filters, "negative_constraints"),
        *feedback_negative_constraints,
    ]
    not_inferred = _feedback_list(filters, "not_inferred")
    validation_priority = _feedback_list(filters, "validation_priority")

    # Backward-compatible fallback if older filter parser still returns the previous schema.
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
    candidate_label = "candidate paper" if candidate_count == 1 else "candidate papers"

    text = f"""Return only one JSON object. No markdown. No explanation.

You are validating the current batch of academic search results semantically.

The current batch contains exactly {candidate_count} {candidate_label}.

User search intent:
{context.user_query}

Current revised topic:
{revised_topic}

Feedback positive constraints:
{_format_items(positive_constraints)}

Feedback negative constraints:
{_format_items(feedback_negative_constraints)}

Query strategy:
{query_strategy}

Planned semantic filters:
Primary intent: {primary_intent}
Conservative filters: {_format_items(conservative_filters)}
Expansive filters: {_format_items(expansive_filters)}
Negative constraints: {_format_items(negative_constraints)}
Not inferred: {_format_items(not_inferred)}
Validation priority: {_format_items(validation_priority)}

Current candidate batch:
{paper_block}

Task:
Validate only the {candidate_count} {candidate_label} in the current candidate batch.
For each candidate, decide whether it is relevant to the user's search intent.

Critical output rule:
- Return exactly {candidate_count} validation object(s) inside validated_papers.
- Return exactly one validation object for each candidate in the current batch.
- Do not skip any candidate.
- Do not add papers that are not in the current candidate batch.
- The paper_id must exactly match the candidate paper id.
- Do not use title, DOI, URL, index, or generated identifiers as paper_id.

Core validation principle:
Be evidence-based and calibrated.
A paper should be included only when the title or abstract supports a real relationship to the user's intent.
Use high for direct matches, medium for strong partial matches, low for useful background, and reject for wrong-topic papers.
Do not include a paper because it merely shares a broad or generic word with the query.

Before validating each paper:
1. Identify the central topic of the user's search.
2. Identify the central task, modality, domain, method, or application required by the filters.
3. Compare the candidate title and abstract against those central requirements.
4. Decide whether the paper is directly relevant, partially relevant, useful background, or irrelevant.

Mandatory gates for direct relevance:
Before assigning high or medium, check:
1. The paper matches the primary_intent when primary_intent is not "none".
2. The paper satisfies the conservative_filters.
3. The paper does not violate negative_constraints.
4. The paper does not depend on assumptions listed in not_inferred.
5. If the topic is computational, the paper has a clear computational contribution.

If a paper fails a direct relevance gate:
- Do not assign high.
- Assign medium only if the missing part is secondary and the paper is still strongly related.
- Assign low/include only if the paper is clearly useful background through a related method, modality, data type, or task family.
- Assign reject/exclude if the paper is wrong topic, wrong domain, wrong modality, wrong task, or only shares generic words.

Conservative vs expansive filter rule:
- Conservative filters are central criteria for direct relevance.
- Expansive filters are only supporting evidence.
- Do not include a paper only because it matches an expansive filter.
- Expansive filters can support low/background relevance when the paper is genuinely related.
- Expansive filters cannot rescue a paper from a clearly wrong domain, wrong modality, or wrong task.
- Negative constraints require exclusion when clearly matched.

Generic words do not prove relevance:
Words such as classification, detection, recognition, prediction, model, models, deep learning, machine learning, artificial intelligence, techniques, methods, trends, survey, review, algorithm, algorithms, framework, system, current, modern, state of the art, and SOTA are generic.
They only count as evidence when they are clearly connected to the central topic of the user's search.

Evidence rule:
- relevance_reason must be specific to the candidate paper being evaluated.
- relevance_reason must cite concrete evidence from that candidate's title or abstract.
- Do not reuse wording from examples.
- Do not copy the same justification across different papers.
- Do not use generic phrases such as "matching the user's search intent" unless you also explain what specific title/abstract evidence matches.
- Do not say "the title explicitly mentions X" unless X or a very close equivalent is actually present in the candidate title.
- If the title/abstract does not support the reason, set relevance="reject" and decision="exclude", unless the paper is clearly useful background and the reason honestly states that weaker relationship.
- Use short paraphrases, not long quotes.
- Do not invent information absent from the title or abstract.
- If the abstract is missing, judge cautiously using title and metadata.
- Every included paper must have a relevance_reason that would still make sense if read without seeing the query.

Reason quality requirements:
Good relevance_reason:
- names the specific concept from the title or abstract that supports relevance;
- explains whether the paper is directly relevant or only useful background;
- does not exaggerate the match.

Bad relevance_reason:
- says the title mentions a topic that the title does not mention;
- repeats the same sentence for unrelated papers;
- relies only on generic terms such as "classification", "deep learning", "model", "techniques", or "survey";
- claims the paper matches the user's intent without explaining why.

Computer Science guidance:
A computational contribution may be an algorithm, model, method, system, architecture, pipeline, software, dataset, benchmark, metric, experiment, performance comparison, technical evaluation, survey, or taxonomy.
Exclude papers that are mainly legal, political, sociological, philosophical, historical, medical, business, conceptual, ethical, or social unless they contain a concrete computational contribution and match the user's intent.

Relevance labels:
- high: directly related to the revised topic and useful for the user's intent.
- medium: strongly related and useful, but missing one secondary aspect.
- low: not directly about the exact topic, but clearly useful as background for methods, concepts, datasets, evaluation, modality, or task family.
- reject: wrong topic, wrong domain, wrong modality, wrong task, unsupported evidence, or violates constraints.

Decision labels:
- include: use for high and medium papers.
- include: use for low only if it is clearly useful background.
- exclude: use for reject.

Output language:
- relevance_reason, mismatch_reason, useful_for, summary, and internal reason text must be in English.
- JSON keys must be in English.

Examples:
The examples below teach the reasoning pattern only.
Do not copy their wording.
Do not copy their topic.
Always adapt the decision and reason to the current candidate title and abstract.

User search intent: explainable AI methods for machine learning models
Conservative filters: explainable artificial intelligence; machine learning interpretability
Candidate title: SHAP-Based Explanations for Tree-Based Machine Learning Models
Expected:
{{"relevance":"high","decision":"include","relevance_reason":"The title directly addresses SHAP explanations for tree-based machine learning models, which matches explainable AI and interpretability methods.","mismatch_reason":"","useful_for":"methods"}}

User search intent: explainable AI methods for machine learning models
Conservative filters: explainable artificial intelligence; machine learning interpretability
Candidate title: A Survey of Explainable Artificial Intelligence: Concepts, Taxonomies, Opportunities and Challenges
Expected:
{{"relevance":"high","decision":"include","relevance_reason":"The title is a survey on explainable artificial intelligence, which is directly useful for understanding XAI concepts and methods.","mismatch_reason":"","useful_for":"overview and concepts"}}

User search intent: explainable AI methods for machine learning models
Conservative filters: explainable artificial intelligence; machine learning interpretability
Candidate title: Improving Accuracy of Convolutional Neural Networks for Plant Disease Classification
Expected:
{{"relevance":"reject","decision":"exclude","relevance_reason":"","mismatch_reason":"The paper is about improving CNN accuracy for plant disease classification, not explainability or interpretability.","useful_for":"not useful"}}

User search intent: explainable AI methods for machine learning models
Conservative filters: explainable artificial intelligence; machine learning interpretability
Candidate title: Fairness and Accountability in Automated Decision Systems
Expected:
{{"relevance":"low","decision":"include","relevance_reason":"The paper is not directly about explainability methods, but it is related to responsible AI and may provide background context if the search includes broader AI accountability.","mismatch_reason":"It focuses on fairness and accountability rather than concrete XAI methods.","useful_for":"background"}}

User search intent: explainable AI methods for machine learning models
Conservative filters: explainable artificial intelligence; machine learning interpretability
Candidate title: Quantum Cryptography for Secure Communication Networks
Expected:
{{"relevance":"reject","decision":"exclude","relevance_reason":"","mismatch_reason":"The paper is about quantum cryptography and secure communication, which does not match explainable AI or machine learning interpretability.","useful_for":"not useful"}}

User search intent: shopping agents using LLMs and VLMs
Conservative filters: shopping agents; LLM or VLM agent; online shopping or purchasing assistance
Candidate title: Web-Based Autonomous Agents for Product Search and Price Comparison
Expected:
{{"relevance":"high","decision":"include","relevance_reason":"The title describes autonomous agents for product search and price comparison, which directly matches shopping or purchasing assistance.","mismatch_reason":"","useful_for":"systems and methods"}}

User search intent: shopping agents using LLMs and VLMs
Conservative filters: shopping agents; LLM or VLM agent; online shopping or purchasing assistance
Candidate title: Deep Generative Modelling: A Comparative Review of VAEs, GANs, Normalizing Flows, Energy-Based and Autoregressive Models
Expected:
{{"relevance":"reject","decision":"exclude","relevance_reason":"","mismatch_reason":"The paper is a general review of generative models and does not show evidence of shopping agents, purchasing assistance, LLM agents, or VLM agents.","useful_for":"not useful"}}

User search intent: bias in medical image classification
Conservative filters: bias; medical image classification
Candidate title: Shortcut Learning in Deep Neural Networks for Medical Image Classification
Expected:
{{"relevance":"high","decision":"include","relevance_reason":"The title connects shortcut learning with medical image classification, which is directly relevant to bias-related model behavior in medical imaging.","mismatch_reason":"","useful_for":"concepts and methods"}}

User search intent: bias in medical image classification
Conservative filters: bias; medical image classification
Candidate title: Generalization in Natural Image Classification Benchmarks
Expected:
{{"relevance":"low","decision":"include","relevance_reason":"The paper is not about medical image classification, but it may provide background on generalization issues in image classification if the search allows broader bias-related context.","mismatch_reason":"It lacks the medical imaging context required for direct relevance.","useful_for":"background"}}

Required JSON shape:
{{"validated_papers":[{{"paper_id":"paper id","relevance":"high","decision":"include","relevance_reason":"short evidence-based reason","mismatch_reason":"","useful_for":"how this helps the research"}}],"summary":"short summary of included and rejected paper types"}}
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
    """Prompt for auditing semantic paper validations."""

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

    text = f"""Return only one JSON object. No markdown. No explanation.

You are an independent AI-as-a-judge auditing paper validations produced by another model.
The original validation is not trustworthy by default.

User search intent:
{context.user_query}

Current revised topic:
{revised_topic}

Search filters:
Primary intent: {primary_intent}
Conservative filters: {_format_items(conservative_filters)}
Expansive filters: {_format_items(expansive_filters)}
Negative constraints: {_format_items(negative_constraints)}

Candidate papers with the only allowed evidence:
{_format_candidate_papers_for_judge(papers)}

Original validations to audit:
{_format_validated_papers_for_judge(validated_papers)}

Task:
Audit exactly {candidate_count} candidate paper(s), one object per paper_id.
Check whether each original decision is correct and whether its reason is supported by the title or abstract.
Correct relevance and decision whenever necessary.

Important:
Before judging individual papers, internally identify the core requirements of the search.
Do not output those requirements separately.
Use them only to audit the original validations.

How to identify core requirements:
Extract the main required parts from:
1. the user search intent;
2. the revised topic;
3. the primary intent;
4. the conservative filters;
5. the expansive filters;
6. the negative constraints.

The core requirements usually include:
- the central subject or domain of the search;
- the central task or problem;
- the required modality, data type, method, or application, when present;
- the type of contribution expected, such as method, model, algorithm, system, survey, benchmark, dataset, experiment, or technical evaluation.

Core judging principle:
Judge the paper, not the previous model's confidence.
The previous validation may be wrong, exaggerated, copied from examples, or unsupported.
Your job is to correct it.

Relevance calibration:
- high: the paper directly satisfies the central topic and the main requirements.
- medium: the paper is strongly related to the central topic, but misses one secondary detail.
- low: the paper does not fully match the central topic, but shares a useful methodological, modality, data-type, or task-family connection that may help as background.
- reject: the paper is from the wrong domain, wrong modality, wrong task, violates negative constraints, or only shares generic words.

Do not be binary:
A paper that is not directly about the exact target topic should not always be rejected.
If it is not directly relevant but is methodologically useful and close to the same modality or task family, use corrected_relevance="low" and corrected_decision="include".

Direct inclusion rule:
Use high or medium only when the paper substantially satisfies the central search intent.

Background inclusion rule:
A paper may be corrected to corrected_relevance="low" and corrected_decision="include" when:
- it does not fully match the exact central topic;
- but it shares the same required modality, data type, or computational task family;
- and it studies a related method, model, recognition/classification problem, detection problem, or technical approach;
- and it does not violate a strong negative constraint.

Strict rejection rule:
Use corrected_relevance="reject" and corrected_decision="exclude" when:
- the paper uses the wrong modality or wrong domain;
- the paper is about a clearly unrelated task;
- the paper violates a negative constraint;
- the paper only matches generic words;
- the paper has no clear methodological, modality, or task-family usefulness for the user topic.

Generic words do not prove relevance:
Words such as classification, detection, recognition, model, models, deep learning, machine learning, artificial intelligence, techniques, methods, trends, survey, review, algorithm, algorithms, current, modern, state of the art, SOTA, system, and framework are generic.
They only count as evidence when they are clearly connected to the central topic of the search.

Conservative and expansive filters:
- Conservative filters are strong evidence for direct relevance.
- A paper that satisfies the conservative filters may be high or medium.
- A paper that misses one exact conservative filter may still be low/include when it is useful background through the same modality, data type, or task family.
- Expansive filters are supporting evidence only.
- Expansive filters can support low/background inclusion, but cannot rescue a paper from a clearly wrong domain or wrong modality.
- Negative constraints are stronger than expansive filters and usually require exclusion.

Reason audit:
- Check whether the original relevance_reason is actually supported by the title or abstract.
- If the original reason says "the title explicitly mentions X", verify the actual title.
- If X or a clear semantic equivalent is not present in the title, reason_is_supported=false.
- If the reason claims evidence that is not in the title or abstract, reason_is_supported=false.
- If the original reason uses the same generic sentence for unrelated papers, reason_is_supported=false.
- If the original reason says the paper matches the user's intent but only mentions generic words, reason_is_supported=false.
- If the original reason overstates relevance, do not automatically reject the paper.
- Instead, correct the relevance level and write a more accurate judge_reason.
- Do not invent evidence.
- Use only the supplied title and abstract.
- If the abstract is absent, use the title cautiously. Reject unless the title clearly supports direct or background relevance.

Invented evidence rule:
If the original reason claims that a title explicitly mentions a topic, but the actual title does not mention that topic or a close equivalent, the original validation is incorrect.
In that case:
- validation_is_correct=false;
- reason_is_supported=false;
- corrected_relevance must be lower than the original relevance;
- corrected_decision must be corrected according to the real evidence.

Repeated reason rule:
If many papers have the same or nearly the same relevance_reason, audit each one independently.
Do not accept repeated justifications unless each specific title or abstract truly supports that justification.

For audio-based violence detection topics:
If the search is about audio-based violence detection:
- Direct papers about violence detection using audio signals should be high/include.
- Papers about audio-based danger detection, aggression detection, scream detection, abnormal sound detection, violent event detection, or security-related acoustic event detection may be high or medium depending on evidence.
- Papers about audio emotion recognition, speech emotion recognition, environmental sound classification, audio event detection, acoustic scene classification, or general audio recognition may be low/include as background methods when audio or speech is central.
- Papers about video-only violence detection, image-only violence detection, visual surveillance, multimodal violence detection where audio is not central, deepfake detection, text-based hate speech, education, agriculture, robotics, drone detection, or unrelated domains should be reject/exclude.

For audio classification topics:
If the search is about audio classification, modern audio classification techniques, or SOTA audio classification:
- audio, sound, speech, acoustic signals, music, voice, environmental sound, or another sonic/audio signal must be central for direct or background inclusion;
- classification, recognition, categorization, identification, detection, or a closely related audio analysis task should be central;
- a computational method, model, algorithm, survey, benchmark, dataset, representation, or technical comparison should be connected to the audio task.
- Reject papers about grape leaf classification, plant disease classification, bacterial classification, generic classification methods without audio, generic deep learning reviews without audio, generic time-series classification without audio, image classification without audio, object detection in images, mixed reality without audio classification, recommendation systems without audio classification, robotics without audio classification, drone detection without audio classification, or education technology without audio classification.

Examples:

Direct positive example:
User topic: audio-based violence detection
Paper title: A novel tree pattern-based violence detection model using audio signals
Original validation: high/include
Expected audit:
- validation_is_correct=true
- reason_is_supported=true
- passes_conservative_filters=true
- violates_negative_constraints=false
- corrected_relevance="high"
- corrected_decision="include"
- judge_reason: "The title directly addresses violence detection using audio signals."

Related positive example:
User topic: audio-based violence detection
Paper title: Audio signal based danger detection using signal processing and deep learning
Original validation: high/include
Expected audit:
- validation_is_correct=true
- reason_is_supported=true
- passes_conservative_filters=true
- violates_negative_constraints=false
- corrected_relevance="high"
- corrected_decision="include"
- judge_reason: "The title describes danger detection from audio signals using signal processing and deep learning, which is closely aligned with audio-based violence or threat detection."

Background example 1:
User topic: audio-based violence detection
Paper title: Supervised machine learning for audio emotion recognition
Original validation: high/include
Expected audit:
- validation_is_correct=false
- reason_is_supported=false
- passes_conservative_filters=false
- violates_negative_constraints=false
- corrected_relevance="low"
- corrected_decision="include"
- judge_reason: "The paper is not about violence detection, but it studies supervised machine learning for audio emotion recognition, which may provide background methods for audio-based recognition."

Background example 2:
User topic: audio-based violence detection
Paper title: Clustering-Based Speech Emotion Recognition by Incorporating Learned Features and Deep BiLSTM
Original validation: high/include
Expected audit:
- validation_is_correct=false
- reason_is_supported=false
- passes_conservative_filters=false
- violates_negative_constraints=false
- corrected_relevance="low"
- corrected_decision="include"
- judge_reason: "The paper focuses on speech emotion recognition rather than violence detection, but it uses audio/speech recognition methods that may be useful as background."

Reject example 1:
User topic: audio-based violence detection
Paper title: A Survey on the Detection and Impacts of Deepfakes in Visual, Audio, and Textual Formats
Original validation: high/include
Original reason: The title explicitly mentions audio-based violence detection.
Expected audit:
- validation_is_correct=false
- reason_is_supported=false
- passes_conservative_filters=false
- violates_negative_constraints=true
- corrected_relevance="reject"
- corrected_decision="exclude"
- judge_reason: "The paper is about deepfake detection across visual, audio, and textual formats, not audio-based violence detection. The original reason invents evidence not present in the title."

Reject example 2:
User topic: audio-based violence detection
Paper title: Deepfake Generation and Detection: Case Study and Challenges
Original validation: high/include
Original reason: The title directly mentions audio-based violence detection.
Expected audit:
- validation_is_correct=false
- reason_is_supported=false
- passes_conservative_filters=false
- violates_negative_constraints=true
- corrected_relevance="reject"
- corrected_decision="exclude"
- judge_reason: "The paper is about deepfake generation and detection, not audio-based violence detection. The original reason is not supported by the title."

Reject example 3:
User topic: audio-based violence detection
Paper title: Advances and Challenges in Drone Detection and Classification Techniques: A State-of-the-Art Review
Original validation: high/include
Original reason: The title explicitly mentions audio-based violence detection.
Expected audit:
- validation_is_correct=false
- reason_is_supported=false
- passes_conservative_filters=false
- violates_negative_constraints=true
- corrected_relevance="reject"
- corrected_decision="exclude"
- judge_reason: "The paper is about drone detection and classification, not audio-based violence detection. The original reason invents evidence not present in the title."

Reject example 4:
User topic: audio-based violence detection
Paper title: Violence Detection in Videos by Combining 3D Convolutional Neural Networks and Support Vector Machines
Original validation: high/include
Expected audit:
- validation_is_correct=false
- reason_is_supported=false
- passes_conservative_filters=false
- violates_negative_constraints=true
- corrected_relevance="reject"
- corrected_decision="exclude"
- judge_reason: "The paper is about video-based violence detection, not audio-based violence detection."

Reject example 5:
User topic: modern audio classification techniques
Paper title: Advancements in deep learning for accurate classification of grape leaves and diagnosis of grape diseases
Original validation: high/include
Expected audit:
- validation_is_correct=false
- reason_is_supported=false
- passes_conservative_filters=false
- violates_negative_constraints=true
- corrected_relevance="reject"
- corrected_decision="exclude"
- judge_reason: "The paper is about grape leaf disease classification, not audio classification."

Reject example 6:
User topic: modern audio classification techniques
Paper title: Deep learning modelling techniques: current progress, applications, advantages, and challenges
Original validation: high/include
Expected audit:
- validation_is_correct=false
- reason_is_supported=false
- passes_conservative_filters=false
- violates_negative_constraints=false
- corrected_relevance="reject"
- corrected_decision="exclude"
- judge_reason: "The paper is a generic deep learning review and does not establish audio classification as the central topic."

Reject example 7:
User topic: modern audio classification techniques
Paper title: A History of Audio Effects
Original validation: high/include
Expected audit:
- validation_is_correct=false
- reason_is_supported=true
- passes_conservative_filters=false
- violates_negative_constraints=false
- corrected_relevance="reject"
- corrected_decision="exclude"
- judge_reason: "The paper is audio-related, but it is about audio effects history, not audio classification techniques."

Positive example:
User topic: modern audio classification techniques
Paper title: Audio classification using self-supervised learning representations
Original validation: high/include
Expected audit:
- validation_is_correct=true
- reason_is_supported=true
- passes_conservative_filters=true
- violates_negative_constraints=false
- corrected_relevance="high"
- corrected_decision="include"
- judge_reason: "The paper directly addresses audio classification using a modern computational technique."

Mandatory judging rules:
- Evaluate semantic subject alignment, not isolated keyword overlap.
- Do not preserve the original decision out of politeness.
- validation_is_correct is true only when the original relevance and decision are both correct.
- passes_conservative_filters is true only when the paper satisfies the direct central conservative requirements.
- passes_conservative_filters may be false for low/include background papers.
- violates_negative_constraints is true when the paper clearly matches a negative constraint or a wrong-domain interpretation.
- If violates_negative_constraints=true, corrected_relevance must be "reject" and corrected_decision must be "exclude".
- judge_reason must be a short, concrete, evidence-based reason in English.

Allowed labels:
- corrected_relevance: "high", "medium", "low", or "reject"
- corrected_decision: "include" or "exclude"

Critical output rules:
- Return exactly {candidate_count} object(s) inside judged_validations.
- Return one object for every candidate paper and no other papers.
- paper_id must exactly match the supplied candidate id.
- All boolean fields must be JSON booleans.
- Do not add extra keys.

Required JSON shape:
{{"judged_validations":[{{"paper_id":"paper id","validation_is_correct":true,"reason_is_supported":true,"passes_conservative_filters":true,"violates_negative_constraints":false,"corrected_relevance":"high","corrected_decision":"include","judge_reason":"short reason"}}],"summary":"short summary"}}
"""

    return PromptSpec(
        name="judge_paper_validations",
        version=JUDGE_PAPER_VALIDATIONS_PROMPT_VERSION,
        text=text,
        metadata={
            "output_format": "json",
            "purpose": "paper_validation_audit",
        },
    )

def build_refine_queries_prompt(
    context: SearchContext,
    validated_papers: list[dict[str, object]],
    used_queries: list[str],
    search_feedback: dict[str, object] | None = None,
) -> PromptSpec:
    """Prompt for refining search queries after feedback and validation."""

    evidence = _format_validated_papers(validated_papers[:12])
    used = "; ".join(used_queries[-8:]) or "none"
    feedback = search_feedback or {}
    revised_topic = _feedback_text(feedback, "revised_topic") or context.user_query
    positive_constraints = _feedback_list(feedback, "positive_constraints")
    negative_constraints = _feedback_list(feedback, "negative_constraints")
    query_strategy = _feedback_text(feedback, "query_strategy") or "none"

    text = f"""Return only one JSON object. No markdown. No explanation.

You are planning a new round of academic search queries after previous results were semantically validated.

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
Create exactly 3 new academic search queries.
The new queries must improve recall while avoiding the mistakes found in rejected papers.
Do not repeat used queries.

Core refinement strategy:
- Use included high/medium papers as evidence for useful terminology.
- Use rejected papers and mismatch reasons as evidence for what to avoid.
- If previous results drifted to the wrong domain, enforce the correct domain.
- If previous results drifted to the wrong modality, enforce the correct modality or close synonym.
- If previous results drifted to the wrong task, enforce the correct task.
- If previous results were too narrow, explore close synonyms or broader faithful terminology.
- If previous results were too broad, use more precise academic terms.
- If the user gave feedback, follow it more strongly than previous evidence.

Core anchors and optional terms:
- Identify core anchors from the revised topic and feedback.
- Core anchors must appear in every query, exactly or as close synonyms.
- Optional technical terms should appear in some queries, not necessarily all.
- Do not make all three queries minor variations of each other.

General-topic behavior:
If the revised topic is general or broad:
- one query may include "state of the art", "trends", "overview", "survey", or "review" when useful;
- do not put those terms in all queries unless the user explicitly asked only for reviews or surveys.

Rules:
- Return only the final JSON object.
- The queries array must contain exactly 3 non-empty strings.
- Academic search queries must be in English.
- The reason must be in English because it is internal.
- Every query must follow the revised topic.
- Every query must respect positive constraints, negative constraints, and query strategy.
- Avoid terms from negative constraints.
- Avoid broad queries that caused irrelevant results before.
- Do not invent datasets, metrics, domains, modalities, methods, applications, or review intent.
- Do not use rejected paper titles as positive query inspiration.

Examples:

Case:
Revised topic: general audio-based violence detection
Rejected evidence: video-based violence detection; hate speech detection; deepfake detection
Output:
{{"queries":[
  "audio-based violence detection",
  "acoustic violent event detection",
  "state of the art sound-based violence detection"
],"reason":"the new queries preserve audio and violence detection anchors while avoiding visual, text, and deepfake drift"}}

Case:
Revised topic: shopping agents using LLMs and VLMs
Rejected evidence: warehouse automation; VR shopping without agents; generic generative AI reviews
Output:
{{"queries":[
  "AI shopping agents",
  "LLM agents for online shopping",
  "VLM shopping assistants in e-commerce"
],"reason":"the new queries preserve shopping agents while avoiding logistics and generic AI drift"}}

Case:
Revised topic: bias in medical image classification
Rejected evidence: generic medical imaging; fairness essays without computational evaluation
Output:
{{"queries":[
  "bias in medical image classification",
  "dataset bias in medical imaging classification",
  "shortcut learning in medical image classification"
],"reason":"the new queries preserve bias and medical image classification while exploring related computational bias terminology"}}

Required JSON shape:
{{"queries":["academic search query 1","academic search query 2","academic search query 3"],"reason":"short reason"}}
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
    """Prompt for interpreting human feedback on validated papers."""

    papers = _format_validated_papers(validated_papers[:20])
    used = "; ".join(used_queries[-10:]) or "none"

    text = f"""Return only one JSON object. No markdown. No explanation.

You are improving an academic paper search after the user reviewed the validated papers.

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
- Keep JSON keys in English.
- Preserve the original research intent unless the user explicitly changes it.
- Put desired concepts, modalities, methods, domains, or terms in positive_constraints.
- Put excluded concepts, modalities, methods, domains, or interpretations in negative_constraints.
- Make query_strategy an actionable sentence for the next query-planning step.
- Keep each constraint short and searchable.
- Use included and excluded papers as evidence for what worked and what failed.
- If the user says a term was ambiguous, choose the clarified meaning and exclude the wrong meanings.

Expected case:
Original user query: audio violence detection
Current refined query: general audio-based violence detection
User feedback: na verdade era para ser apenas audio, sem multimodal e sem audio-visual
Output:
{{"revised_topic":"general audio-only violence detection","positive_constraints":["audio-only violence detection","sound-based detection","acoustic signals"],"negative_constraints":["multimodal","audio-visual","audiovisual","video-based detection","visual surveillance"],"query_strategy":"Search for violence detection papers where audio, sound, or acoustic signals are the central input. Avoid multimodal, audiovisual, video-based, or visual-surveillance papers.","reason":"The user clarified that the intended modality is audio only and rejected multimodal or audiovisual results."}}

Required JSON shape:
{{"revised_topic":"clean revised academic topic","positive_constraints":["constraint"],"negative_constraints":["constraint"],"query_strategy":"actionable query strategy","reason":"short reason"}}
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
    """Prompt for deciding whether another search round is useful."""

    evidence = _format_validated_papers(validated_papers[:10])

    text = f"""Return only one JSON object. No markdown. No explanation.

Decide if another academic search round is useful.

Topic: {context.user_query}
Round: {round_number} of {context.max_rounds}
New papers: {last_new_paper_count}
New useful validated papers: {last_new_useful_count}
Validated-paper evidence:
{evidence}

Decision rules:
- Continue if there are too few useful high/medium papers and the rejected evidence suggests a fixable query drift.
- Stop if enough useful validated papers were found.
- Stop if the latest round mostly repeats old mistakes and no clear new query direction exists.
- The reason must be in English because it is internal.

Required JSON shape:
{{"continue":true,"reason":"short reason"}}
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
    return "; ".join(items) if items else "none"


def _format_candidate_papers(papers: list[Paper]) -> str:
    if not papers:
        return "none"

    lines: list[str] = []
    for position, paper in enumerate(papers, start=1):
        abstract = " ".join((paper.abstract or "").split())[:700] or "no abstract"
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


def _format_validated_papers_for_judge(
    validated_papers: list[dict[str, object]],
) -> str:
    if not validated_papers:
        return "none"

    lines: list[str] = []
    for position, item in enumerate(validated_papers, start=1):
        paper = item.get("paper")
        paper_id = getattr(paper, "id", item.get("paper_id", "unknown"))
        lines.append(
            f"{position}. paper_id={paper_id}; "
            f"relevance={_feedback_text(item, 'relevance') or 'unknown'}; "
            f"decision={_feedback_text(item, 'decision') or 'unknown'}; "
            f"relevance_reason={_feedback_text(item, 'relevance_reason') or 'none'}; "
            f"mismatch_reason={_feedback_text(item, 'mismatch_reason') or 'none'}; "
            f"useful_for={_feedback_text(item, 'useful_for') or 'none'}"
        )
    return "\n".join(lines)


def _format_candidate_papers_for_judge(papers: list[Paper]) -> str:
    if not papers:
        return "none"

    lines: list[str] = []
    for position, paper in enumerate(papers, start=1):
        abstract = " ".join((paper.abstract or "").split())[:700] or "no abstract"
        lines.append(
            f"{position}. paper_id={paper.id}; title={paper.title}; abstract={abstract}"
        )
    return "\n".join(lines)
