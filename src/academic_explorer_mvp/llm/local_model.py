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
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

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

            self._model = model
            self._tokenizer = tokenizer
            self._torch = torch

        except Exception as exc:
            raise LocalModelError(
                "Could not load the configured local model "
                f"{config.agent_model_id!r} from ACADEMIC_EXPLORER_AGENT_MODEL_ID. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

    def generate(self, prompt: str) -> str:
        """Generate text from the local model."""

        formatted_prompt = self._format_prompt(prompt)
        inputs = self._tokenizer(formatted_prompt, return_tensors="pt")
        inputs = self._move_inputs_to_model_device(inputs)
        input_token_count = inputs["input_ids"].shape[-1]

        generation_args: dict[str, Any] = {
            "max_new_tokens": self.config.agent_max_new_tokens,
            "do_sample": self.config.agent_temperature > 0,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if self.config.agent_temperature > 0:
            generation_args["temperature"] = self.config.agent_temperature

        try:
            with self._torch.inference_mode():
                output_ids = self._model.generate(**inputs, **generation_args)
        except Exception as exc:
            raise LocalModelError(
                "Local model generation failed. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        if output_ids is None or len(output_ids) == 0:
            raise LocalModelError("Local model returned no output.")

        new_token_ids = output_ids[0][input_token_count:]
        text = self._tokenizer.decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        if not text.strip():
            raise LocalModelError("Local model returned empty text.")

        return text

    def _move_inputs_to_model_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        device = self._model_device()
        if device is None:
            return inputs

        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def _model_device(self) -> Any | None:
        device = getattr(self._model, "device", None)
        if device is not None and str(device) != "meta":
            return device

        try:
            parameter = next(self._model.parameters())
        except StopIteration:
            return None

        device = getattr(parameter, "device", None)
        if device is None or str(device) == "meta":
            return None
        return device

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
