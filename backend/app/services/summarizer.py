import asyncio
import json
import time
from typing import Optional, AsyncGenerator, Any
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import sanitize_document_text, format_data_boundary
from backend.app.db.models import Document, SummaryResult
from backend.app.db.schemas import (
    SummarizeRequest, SummarizeResponse, StudyPackageOutput, PageChunk
)
from backend.app.prompts.templates import (
    PROMPT_B_STRUCTURED_SYSTEM, PROMPT_B_STRUCTURED_USER,
    PROMPT_A_DIRECT_SYSTEM, PROMPT_A_DIRECT_USER, render_prompt
)
from backend.app.providers.factory import ProviderFactory
from backend.app.providers.offline_provider import OfflineProvider
from backend.app.services.chunker import MapReduceChunker

class SummarizerService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def summarize(
        self,
        request: SummarizeRequest,
        api_key: Optional[str] = None
    ) -> SummarizeResponse:
        start_time = time.time()
        
        # 1. Fetch document text
        doc_title = request.title or "Lecture Notes"
        full_text = ""
        pages_data = []
        doc_id = request.document_id

        if doc_id:
            db_doc = await self.db.get(Document, doc_id)
            if not db_doc:
                raise ValueError(f"Document with ID '{doc_id}' not found.")
            full_text = db_doc.full_text
            doc_title = db_doc.title
            try:
                pages_data = [PageChunk.model_validate(p) for p in json.loads(db_doc.pages_data)]
            except Exception:
                pages_data = []
        elif request.raw_text:
            full_text = request.raw_text
            pages_data = [PageChunk(page_number=1, title="Notes", text=full_text)]
        else:
            raise ValueError("Either document_id or raw_text must be provided.")

        # Sanitize and format data boundaries
        sanitized_text, _ = sanitize_document_text(full_text)
        if not sanitized_text:
            raise ValueError("Document contains no valid text to summarize.")

        # 2. Configure target word count
        if request.summary_mode == "short":
            target_words = 130
        elif request.summary_mode == "detailed":
            target_words = 550
        elif request.target_words:
            target_words = request.target_words
        else:
            target_words = 250

        # 3. Choose provider (Gemini or Offline)
        provider = ProviderFactory.get_provider(api_key=api_key)
        provider_type = "gemini" if ProviderFactory.is_gemini_active(api_key) else "offline"
        model_name = request.model or ("gemini-3.5-flash-lite" if provider_type == "gemini" else "offline-extractive-v1")
        temp = request.temperature or 0.2

        # 4. Check if map-reduce chunking is required
        total_words = len(sanitized_text.split())
        needs_map_reduce = MapReduceChunker.should_use_map_reduce(pages_data, total_words)

        try:
            if needs_map_reduce and len(pages_data) > 1:
                study_pkg = await self._run_map_reduce(
                    pages_data=pages_data,
                    title=doc_title,
                    target_words=target_words,
                    audience=request.audience_level,
                    language=request.language,
                    provider=provider,
                    temperature=temp,
                    model_name=model_name
                )
            else:
                # Single-pass generation
                variables = {
                    "title": doc_title,
                    "notes": format_data_boundary(sanitized_text),
                    "length": target_words,
                    "audience": request.audience_level,
                    "language": request.language,
                }
                # For primary summarizer, use Structured cognitive scaffold for highest quality
                system_prompt = render_prompt(PROMPT_B_STRUCTURED_SYSTEM, variables)
                user_prompt = render_prompt(PROMPT_B_STRUCTURED_USER, variables)
                
                study_pkg, _, _ = await provider.generate_study_package(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=temp,
                    model_name=model_name
                )
        except Exception as e:
            if provider_type == "gemini":
                # Resilient fallback to offline extractive engine so users are never blocked
                offline_prov = OfflineProvider()
                study_pkg, _, _ = await offline_prov.generate_study_package(
                    user_prompt=sanitized_text,
                    system_prompt="",
                    temperature=temp,
                    model_name="offline-extractive-v1"
                )
                provider_type = "offline"
                model_name = "offline-extractive-v1"
            else:
                raise e

        # 5. Apply toggles
        if not request.generate_summary:
            study_pkg.summary = ""
        if not request.generate_concepts:
            study_pkg.key_concepts = []
        if not request.generate_revision_points:
            study_pkg.revision_points = []
        if not request.generate_flashcards:
            study_pkg.flashcards = []

        total_latency = round((time.time() - start_time) * 1000, 2)

        # 6. Persist result
        summary_record = SummaryResult(
            document_id=doc_id or "adhoc",
            summary_mode=request.summary_mode,
            target_words=target_words,
            audience_level=request.audience_level,
            language=request.language,
            summary_text=study_pkg.summary,
            key_concepts=json.dumps([c.model_dump() for c in study_pkg.key_concepts]),
            revision_points=json.dumps([r.model_dump() for r in study_pkg.revision_points]),
            flashcards=json.dumps([f.model_dump() for f in study_pkg.flashcards]),
            provider_type=provider_type,
            model_name=model_name,
            latency_ms=total_latency
        )
        self.db.add(summary_record)
        await self.db.commit()
        await self.db.refresh(summary_record)

        return SummarizeResponse(
            id=summary_record.id,
            document_id=summary_record.document_id,
            summary_mode=summary_record.summary_mode,
            audience_level=summary_record.audience_level,
            language=summary_record.language,
            output=study_pkg,
            provider_type=provider_type,
            model_name=model_name,
            latency_ms=total_latency,
            created_at=summary_record.created_at
        )

    async def _run_map_reduce(
        self,
        pages_data: list[PageChunk],
        title: str,
        target_words: int,
        audience: str,
        language: str,
        provider: Any,
        temperature: float,
        model_name: str
    ) -> StudyPackageOutput:
        """Executes parallel map summarization across batches, followed by reduce synthesis."""
        batches = MapReduceChunker.partition_pages(pages_data)
        
        # MAP PHASE: Summarize each batch concurrently
        async def process_batch(batch_idx: int, batch_pages: list[PageChunk]) -> StudyPackageOutput:
            batch_text = MapReduceChunker.combine_chunk_texts(batch_pages)
            variables = {
                "title": f"{title} (Part {batch_idx + 1})",
                "notes": format_data_boundary(batch_text),
                "length": 150,
                "audience": audience,
                "language": language,
            }
            sys_prompt = render_prompt(PROMPT_A_DIRECT_SYSTEM, variables)
            usr_prompt = render_prompt(PROMPT_A_DIRECT_USER, variables)
            out, _, _ = await provider.generate_study_package(
                user_prompt=usr_prompt,
                system_prompt=sys_prompt,
                temperature=temperature,
                model_name=model_name
            )
            return out

        map_tasks = [process_batch(i, batch) for i, batch in enumerate(batches)]
        map_results = await asyncio.gather(*map_tasks)

        # REDUCE PHASE: Aggregate intermediate outputs
        aggregated_notes = []
        all_concepts = []
        all_revisions = []
        all_flashcards = []

        for idx, res in enumerate(map_results):
            aggregated_notes.append(f"### Section {idx+1} Key Points\n{res.summary}")
            all_concepts.extend(res.key_concepts)
            all_revisions.extend(res.revision_points)
            all_flashcards.extend(res.flashcards)

        combined_intermediate_text = "\n\n".join(aggregated_notes)

        # Final synthesize reduce pass
        reduce_variables = {
            "title": title,
            "notes": format_data_boundary(combined_intermediate_text),
            "length": target_words,
            "audience": audience,
            "language": language,
        }
        reduce_sys = render_prompt(PROMPT_B_STRUCTURED_SYSTEM, reduce_variables)
        reduce_usr = render_prompt(PROMPT_B_STRUCTURED_USER, reduce_variables)

        final_pkg, _, _ = await provider.generate_study_package(
            user_prompt=reduce_usr,
            system_prompt=reduce_sys,
            temperature=temperature,
            model_name=model_name
        )

        # Merge unique concepts, revision points, and flashcards
        seen_concepts = set()
        merged_concepts = []
        for c in final_pkg.key_concepts + all_concepts:
            k = c.term.strip().lower()
            if k not in seen_concepts:
                seen_concepts.add(k)
                merged_concepts.append(c)

        final_pkg.key_concepts = merged_concepts[:12]
        if not final_pkg.revision_points:
            final_pkg.revision_points = all_revisions[:8]
        if not final_pkg.flashcards:
            final_pkg.flashcards = all_flashcards[:8]

        return final_pkg

    async def stream_summary(
        self,
        request: SummarizeRequest,
        api_key: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Streams summary tokens/chunks using Server-Sent Events."""
        full_text = request.raw_text or ""
        if request.document_id:
            db_doc = await self.db.get(Document, request.document_id)
            if db_doc:
                full_text = db_doc.full_text

        sanitized_text, _ = sanitize_document_text(full_text)
        provider = ProviderFactory.get_provider(api_key=api_key)
        
        target_words = 130 if request.summary_mode == "short" else 500
        variables = {
            "title": request.title or "Lecture Notes",
            "notes": format_data_boundary(sanitized_text),
            "length": target_words,
            "audience": request.audience_level,
            "language": request.language,
        }
        
        system_prompt = render_prompt(PROMPT_B_STRUCTURED_SYSTEM, variables)
        user_prompt = render_prompt(PROMPT_B_STRUCTURED_USER, variables)
        
        async for chunk in provider.stream_summary(user_prompt, system_prompt, request.temperature or 0.2):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"
