"""Thin local model wrapper backed by transformers."""

from importlib.util import find_spec
import json
import re
from typing import Any

from academic_explorer_mvp.config import AppConfig


class LocalModelError(RuntimeError):
    """Configuration or runtime problem with the local model."""


class LocalModelJsonError(LocalModelError):
    """The local model answered, but not with valid JSON."""


try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ModuleNotFoundError as exc:
    missing = exc.name or "transformers"
    raise LocalModelError(
        f"Missing dependency {missing!r} for the local agent. "
        "Install model dependencies with: py -m pip install -e \".[local-model]\". "
        "The model is configured by ACADEMIC_EXPLORER_AGENT_MODEL_ID."
    ) from exc


class LocalModel:
    """Load and call a local text-generation model.

    This class has no deterministic replacement path. If transformers or the
    configured model cannot load, the CLI stops with a clear configuration error.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config

        self._ensure_dependency(
            package_name="transformers",
            env_name="ACADEMIC_EXPLORER_AGENT_MODEL_ID",
        )
        self._ensure_dependency(
            package_name="torch",
            env_name="ACADEMIC_EXPLORER_AGENT_TORCH_DTYPE",
        )

        if config.agent_device_map == "auto":
            self._ensure_dependency(
                package_name="accelerate",
                env_name="ACADEMIC_EXPLORER_AGENT_DEVICE_MAP",
            )

        try:
            tokenizer = AutoTokenizer.from_pretrained(config.agent_model_id)

            model_kwargs: dict[str, Any] = {
                "device_map": config.agent_device_map,
                "torch_dtype": config.agent_torch_dtype,
            }

            model = AutoModelForCausalLM.from_pretrained(
                config.agent_model_id,
                **model_kwargs,
            )

            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token

            self._configure_generation(model, tokenizer)

            self._model = model
            self._tokenizer = tokenizer

            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
            )

        except Exception as exc:
            raise LocalModelError(
                "Could not load the configured local model "
                f"{config.agent_model_id!r} from ACADEMIC_EXPLORER_AGENT_MODEL_ID. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

    def _configure_generation(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
        """Configure generation directly on the model.

        Important:
        - Use max_new_tokens, not max_length.
        - Do not pass generation_config into pipeline().
        - Do not pass max_new_tokens/do_sample/temperature in pipeline calls.
        """

        model.generation_config.max_length = None
        model.generation_config.max_new_tokens = self.config.agent_max_new_tokens

        model.generation_config.do_sample = self.config.agent_temperature > 0

        if self.config.agent_temperature > 0:
            model.generation_config.temperature = self.config.agent_temperature
        else:
            model.generation_config.temperature = None

        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id

    def generate(self, prompt: str) -> str:
        """Generate text from the local model."""

        pipeline_args: dict[str, Any] = {
            "return_full_text": False,
            "clean_up_tokenization_spaces": False,
        }

        try:
            outputs = self._pipeline(
                self._format_prompt(prompt),
                **pipeline_args,
            )
        except Exception as exc:
            raise LocalModelError(
                "Local model generation failed. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        if not outputs:
            raise LocalModelError("Local model returned no output.")

        first = outputs[0]

        if isinstance(first, dict):
            text = str(first.get("generated_text") or "")
        else:
            text = str(first)

        if not text.strip():
            raise LocalModelError("Local model returned empty text.")

        return text

    def _format_prompt(self, prompt: str) -> str:
        if not hasattr(self._tokenizer, "apply_chat_template"):
            return prompt

        messages = [
            {
                "role": "system",
                "content": (
                    "Follow the user instruction exactly. When JSON is requested, "
                    "return only valid JSON and no prose."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    def generate_json(self, prompt: str) -> dict[str, Any] | list[Any]:
        """Generate text and parse a JSON object or list from it."""

        text = self.generate(prompt)
        parsed = self._parse_json(text)

        if parsed is None:
            preview = " ".join(text.split())[:240]
            raise LocalModelJsonError(
                "Local model did not return valid JSON. QueryPlanner expects only a "
                "JSON object or list. Check the prompt/model configuration; raw "
                f"preview: {preview!r}"
            )

        return parsed

    def _ensure_dependency(self, package_name: str, env_name: str) -> None:
        if find_spec(package_name) is not None:
            return

        raise LocalModelError(
            f"Missing dependency {package_name!r} required by the local agent. "
            f"This run uses {env_name}. Install model dependencies with: "
            "py -m pip install -e \".[local-model]\"."
        )

    def _parse_json(self, text: str) -> dict[str, Any] | list[Any] | None:
        cleaned = self._strip_code_fence(text.strip())

        for candidate in self._json_candidates(cleaned):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, (dict, list)):
                return parsed

        return None

    def _strip_code_fence(self, text: str) -> str:
        if text.startswith("```"):
            return re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                text,
                flags=re.IGNORECASE,
            )

        return text

    def _json_candidates(self, text: str) -> list[str]:
        candidates = [text]

        object_start = text.find("{")
        object_end = text.rfind("}")

        if object_start != -1 and object_end > object_start:
            candidates.append(text[object_start : object_end + 1])

        list_start = text.find("[")
        list_end = text.rfind("]")

        if list_start != -1 and list_end > list_start:
            candidates.append(text[list_start : list_end + 1])

        return candidates