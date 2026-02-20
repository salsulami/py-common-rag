"""Default prompt templates for all RAG components."""

PROMPT_QUERY_EXPANSION = "retrieval.query_expansion"
PROMPT_QUERY_DECOMPOSITION = "retrieval.query_decomposition"
PROMPT_HYPOTHESIS = "reasoning.hypothesis_generation"
PROMPT_PDF_CLEANUP = "parsers.pdf.cleanup"
PROMPT_PPTX_CLEANUP = "parsers.pptx.cleanup"
PROMPT_DOCX_CLEANUP = "parsers.docx.cleanup"
PROMPT_PDF_VISION_EXTRACTION = "parsers.pdf.vision_markdown_caption"
PROMPT_PPTX_VISION_EXTRACTION = "parsers.pptx.vision_markdown_caption"
PROMPT_DOCX_VISION_EXTRACTION = "parsers.docx.vision_markdown_caption"

DEFAULT_PROMPTS: dict[str, str] = {
    PROMPT_QUERY_EXPANSION: """
You are an expert retrieval engineer.
Expand the user query into multiple retrieval-ready alternatives.
Preserve intent, improve recall, include synonyms, entities, and constrained reformulations.
Output strict JSON only with this schema:
{{
  "expansions": ["..."],
  "rationale": "..."
}}

Constraints:
- Return {target_count} expansions unless impossible.
- Expansions must be distinct and useful for vector + keyword retrieval.
- Keep each expansion concise and self-contained.

Query: {query}
Context: {context}
""".strip(),
    PROMPT_QUERY_DECOMPOSITION: """
You are an expert planner for multi-hop retrieval.
Break the user query into executable sub-queries with dependencies.
Output strict JSON only with this schema:
{{
  "sub_queries": [
    {{
      "id": "q1",
      "query": "...",
      "intent": "...",
      "depends_on": []
    }}
  ],
  "execution_notes": "..."
}}

Constraints:
- Return at most {max_steps} sub-queries.
- Each sub-query must be independently retrievable.
- Use depends_on to encode ordering when a later query needs prior outputs.

Query: {query}
Context: {context}
""".strip(),
    PROMPT_HYPOTHESIS: """
You are an expert analyst generating evidence-driven hypotheses for RAG.
Given a user query and retrieved context, produce plausible hypotheses and evidence needs.
Output strict JSON only with this schema:
{{
  "hypotheses": [
    {{
      "statement": "...",
      "confidence": 0.0,
      "required_evidence": ["..."],
      "rationale": "..."
    }}
  ],
  "overall_risk": "..."
}}

Constraints:
- Produce exactly {hypothesis_count} hypotheses unless context is insufficient.
- confidence must be between 0 and 1.
- required_evidence should be concrete and verifiable.
- Prefer cautious confidence when evidence is weak.

Query: {query}
Retrieved context:
{context}
""".strip(),
    PROMPT_PDF_CLEANUP: """
You normalize PDF extracted text for downstream retrieval.
Output strict JSON only:
{{
  "clean_text": "..."
}}

Rules:
- Fix obvious OCR artifacts and broken line wraps.
- Preserve factual meaning; do not invent content.
- Keep numbered/bulleted structures readable.

Text:
{text}
""".strip(),
    PROMPT_PPTX_CLEANUP: """
You normalize PowerPoint slide text for downstream retrieval.
Output strict JSON only:
{{
  "clean_text": "..."
}}

Rules:
- Keep key bullet points and headings.
- Remove decorative noise.
- Preserve factual meaning.

Text:
{text}
""".strip(),
    PROMPT_DOCX_CLEANUP: """
You normalize Word document text for downstream retrieval.
Output strict JSON only:
{{
  "clean_text": "..."
}}

Rules:
- Preserve section meaning and business semantics.
- Merge fragmented lines only when safe.
- Preserve factual meaning.

Text:
{text}
""".strip(),
    PROMPT_PDF_VISION_EXTRACTION: """
You are reading a screenshot of a PDF page.
Extract faithful Markdown and create a concise caption.
Output strict JSON only:
{{
  "markdown": "...",
  "caption": "..."
}}

Rules:
- Capture all legible text and structure using Markdown.
- Preserve headings, lists, and table-like layouts when visible.
- Do not invent content that is not present in the image.
- caption should summarize the page in one sentence.

Metadata:
- source_path: {source_path}
- item_type: {item_type}
- index: {index}
""".strip(),
    PROMPT_PPTX_VISION_EXTRACTION: """
You are reading a screenshot of a PowerPoint slide.
Extract faithful Markdown and create a concise caption.
Output strict JSON only:
{{
  "markdown": "...",
  "caption": "..."
}}

Rules:
- Preserve title, bullets, and visible table/text hierarchy.
- Include key chart/diagram labels if legible.
- Do not invent content that is not present in the image.
- caption should summarize the slide in one sentence.

Metadata:
- source_path: {source_path}
- item_type: {item_type}
- index: {index}
""".strip(),
    PROMPT_DOCX_VISION_EXTRACTION: """
You are reading a screenshot of a Word document page.
Extract faithful Markdown and create a concise caption.
Output strict JSON only:
{{
  "markdown": "...",
  "caption": "..."
}}

Rules:
- Preserve visible section hierarchy and list/numbering structure.
- Reconstruct table-like structures when clearly visible.
- Do not invent content that is not present in the image.
- caption should summarize the page in one sentence.

Metadata:
- source_path: {source_path}
- item_type: {item_type}
- index: {index}
""".strip(),
}
