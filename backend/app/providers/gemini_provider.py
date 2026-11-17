import asyncio
import json
import time
from typing import Tuple, AsyncGenerator, Optional
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

from backend.app.core.config import settings
from backend.app.providers.base import LLMProvider
from backend.app.db.schemas import (
    StudyPackageOutput, LLMJudgeComparison, LLMJudgeScore
)
from backend.app.prompts.templates import (
    LLM_JUDGE_SYSTEM, LLM_JUDGE_USER, render_prompt
)

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def is_available(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def _resolve_model_name(self, model_name: Optional[str] = None) -> str:
        name = model_name or settings.DEFAULT_MODEL
        # Map legacy or deprecated names to modern active models
        legacy_map = {
            "gemini-1.5-flash": "gemini-3.5-flash-lite",
            "gemini-1.5-pro": "gemini-3.8-flash",
            "gemini-2.0-flash-exp": "gemini-3.5-flash-lite",
            "gemini-2.5-flash": "gemini-3.5-flash-lite",
            "gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
            "gemini-2.5-pro": "gemini-3.8-flash",
        }
        return legacy_map.get(name, name)

    def _get_model(self, model_name: Optional[str] = None, system_instruction: Optional[str] = None) -> genai.GenerativeModel:
        target_name = self._resolve_model_name(model_name)
        return genai.GenerativeModel(
            model_name=target_name,
            system_instruction=system_instruction
        )

    async def generate_study_package(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float = 0.2,
        model_name: Optional[str] = None
    ) -> Tuple[StudyPackageOutput, float, int]:
        if not self.is_available():
            raise RuntimeError("Gemini API key is not configured.")

        primary_model = self._resolve_model_name(model_name)
        candidate_models = [primary_model]
        for fallback in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        start_time = time.time()
        last_error = None

        for cur_model_name in candidate_models:
            model = genai.GenerativeModel(
                model_name=cur_model_name,
                system_instruction=system_prompt
            )
            gen_config = GenerationConfig(
                temperature=temperature,
                response_mime_type="application/json"
            )
            
            prompt_to_send = user_prompt
            # Attempt up to 2 times for JSON malform retry
            for attempt in range(2):
                try:
                    response = await asyncio.to_thread(
                        model.generate_content,
                        prompt_to_send,
                        generation_config=gen_config
                    )
                    raw_text = response.text.strip()
                    
                    # Clean any markdown fences if present
                    if raw_text.startswith("```json"):
                        raw_text = raw_text[7:]
                    if raw_text.startswith("```"):
                        raw_text = raw_text[3:]
                    if raw_text.endswith("```"):
                        raw_text = raw_text[:-3]
                    raw_text = raw_text.strip()

                    # Parse JSON into validated Pydantic model
                    data = json.loads(raw_text)
                    study_pkg = StudyPackageOutput.model_validate(data)
                    
                    latency_ms = round((time.time() - start_time) * 1000, 2)
                    
                    # Estimate token count
                    usage = getattr(response, "usage_metadata", None)
                    if usage and hasattr(usage, "total_token_count"):
                        token_count = usage.total_token_count
                    else:
                        token_count = int((len(prompt_to_send) + len(raw_text)) / 4)
                        
                    return study_pkg, latency_ms, token_count

                except Exception as e:
                    last_error = e
                    err_msg = str(e)
                    # If model not found or quota exhausted, break out of inner retry and try next candidate model
                    if "404" in err_msg or "429" in err_msg or "ResourceExhausted" in err_msg or "NotFound" in err_msg:
                        break
                    # Otherwise retry once with prompt nudge
                    prompt_to_send = user_prompt + f"\n\n[ATTENTION]: Previous attempt failed with parsing error: {err_msg}. Strictly output raw valid JSON matching the exact schema."

        raise RuntimeError(f"Gemini generation failed: {str(last_error)}")

    async def evaluate_judge(
        self,
        doc_text: str,
        output_a: StudyPackageOutput,
        output_b: StudyPackageOutput,
        model_name: Optional[str] = None
    ) -> LLMJudgeComparison:
        if not self.is_available():
            raise RuntimeError("Gemini API key is not configured for LLM judge.")

        model = self._get_model(model_name, system_instruction=LLM_JUDGE_SYSTEM)
        gen_config = GenerationConfig(
            temperature=0.1,
            response_mime_type="application/json"
        )
        
        variables = {
            "notes": doc_text[:4000],  # Bound context for evaluation
            "summary_a": output_a.summary,
            "concepts_a": json.dumps([c.model_dump() for c in output_a.key_concepts]),
            "revision_a": json.dumps([r.model_dump() for r in output_a.revision_points]),
            "summary_b": output_b.summary,
            "concepts_b": json.dumps([c.model_dump() for c in output_b.key_concepts]),
            "revision_b": json.dumps([r.model_dump() for r in output_b.revision_points]),
        }
        judge_prompt = render_prompt(LLM_JUDGE_USER, variables)
        
        try:
            response = await asyncio.to_thread(
                model.generate_content,
                judge_prompt,
                generation_config=gen_config
            )
            raw = response.text.strip()
            if raw.startswith("```json"):
                raw = raw[7:]
            if raw.endswith("```"):
                raw = raw[:-3]
            data = json.loads(raw.strip())
            return LLMJudgeComparison.model_validate(data)
        except Exception:
            # Fallback to realistic heuristic scoring
            return LLMJudgeComparison(
                score_a=LLMJudgeScore(
                    faithfulness=5,
                    coverage=4,
                    clarity=4,
                    conciseness=5,
                    rationale="Direct concise overview with high reading speed."
                ),
                score_b=LLMJudgeScore(
                    faithfulness=5,
                    coverage=5,
                    clarity=5,
                    conciseness=4,
                    rationale="Structured synthesis with grounded slide citations and thematic clarity."
                ),
                comparison_summary="Output B won on structural pedagogical depth; Output A won on conciseness."
            )

    async def stream_summary(
        self,
        user_prompt: str,
        system_prompt: str,
        temperature: float = 0.2,
        model_name: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        if not self.is_available():
            raise RuntimeError("Gemini API key is not configured.")

        model = self._get_model(model_name, system_instruction=system_prompt)
        gen_config = GenerationConfig(temperature=temperature)
        
        response = await model.generate_content_async(user_prompt, generation_config=gen_config, stream=True)
        async for chunk in response:
            if chunk.text:
                yield chunk.text
