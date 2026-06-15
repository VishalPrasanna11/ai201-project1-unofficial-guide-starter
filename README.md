# The Unofficial Guide — Project 1

A retrieval-augmented generation (RAG) assistant for **Northeastern University Housing and Residential Life**. It answers questions about on-campus housing policies, rates, application deadlines, and student dorm reviews — grounded in a curated document corpus with source attribution.

---

## Domain

**Northeastern University Housing and Residential Life** — on-campus and university-affiliated housing for Northeastern undergraduates in Boston.

The system covers 40+ residential communities (traditional, suite, and apartment styles), two-year housing requirements, application timelines by student type, Living Learning Communities, per-semester rates and meal plans, move-in/out procedures, utilities, all-gender housing, packing rules, and unofficial RoomSurf dorm reviews.

**Why this knowledge is valuable:** Housing is a high-stakes decision for incoming Huskies. Students and families need answers on how selection works, what it costs, what to pack, and what daily life feels like — not just building names. NUin spring returners face unique placement into suites or apartments based on deposit date order, not specific hall preference.

**Why it is hard to find through official channels:** Housing info is spread across Housing Online, PDF rate charts, LLC brochures, and policy pages with frequent updates. Official sources describe buildings but omit lived experience — thin walls, forced doubles, overflow isolation. Students piece together answers from scattered pages, Reddit, and word-of-mouth. This guide unifies those sources into one queryable system.

---

## Document Sources

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | Housing & Residential Life (Home) | Official NU page | https://housing.northeastern.edu/ · `documents/welcome_info.txt` |
| 2 | University Housing | Official NU page | https://housing.northeastern.edu/university-housing/ · `documents/university_housing.txt` |
| 3 | Housing Application & Selection | Official NU page | https://housing.northeastern.edu/applyselect/ · `documents/application_process.txt` |
| 4 | What To Bring | Official NU page | https://housing.northeastern.edu/what-to-bring/ · `documents/what_to_bring.txt` |
| 5 | All Gender Housing | Official NU page | https://housing.northeastern.edu/university-housing/all-gender-housing/ · `documents/all_gender_housing.txt` |
| 6 | Living Learning Communities | Official NU page | https://housing.northeastern.edu/living-learning-communities/ · `documents/Living_Learning_Communities.txt` |
| 7 | NUin Spring Housing | Official NU page | https://housing.northeastern.edu/nuin/ · `documents/spring_housing.txt` |
| 8 | Room Rates (2025–2026) | Official NU page | https://housing.northeastern.edu/room-rates/ · `documents/room_rates.txt` |
| 9 | Residential Utilities | Official NU page | https://housing.northeastern.edu/residential-utilities/ · `documents/residential_utlilies.txt` |
| 10 | Student Dorm Reviews (RoomSurf) | Unofficial reviews | https://www.roomsurf.com/dorm-reviews/neu · `documents/dorm_review.txt` |

Additional sources collected but not listed above: Fall Move-In/Out (`documents/fall_move_out.txt`).

---

## Chunking Strategy

**Chunk size:** 400 tokens (~1,600 characters), hard cap at 512 tokens (~2,048 characters). This fits the embedding model's input limit while keeping enough context for policy paragraphs and rate lines.

**Overlap:** 60 tokens (~240 characters, ~15%) for recursive/policy documents. Overlap helps when a rule or deadline gets cut at a chunk boundary. No overlap for room rates or dorm reviews.

**Why these choices fit your documents:** Most NU housing sources (application process, university housing, LLCs, utilities, NUin) are structured guides with headers and bullet lists — recursive splitting on `\n\n`, then `\n`, then sentences works well. Two files needed special handling:

- **`room_rates.txt`** — one chunk per building (e.g., Kerr Hall + all its rates together). Splitting a building header from its price lines would break rate lookups.
- **`dorm_review.txt`** — one chunk per RoomSurf review. Reviews are short and self-contained; splitting would lose the student voice.

**Preprocessing:** Strip extra whitespace; tag each chunk with source filename and section/building name for citation at retrieval time.

**Final chunk count:** **144 chunks** from 11 documents (104 building-rate chunks, 8 dorm reviews, 32 from recursive split on policy/guide files).

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`. It is lightweight, runs locally with no API cost, and handles short factual/policy text well — which matches most housing chunks. Each chunk is embedded once at ingest time and stored in a persistent ChromaDB collection (`housing_chunks`).

**Production tradeoff reflection:** If cost and latency were not constraints, I would weigh:

- **Accuracy on domain text** — a larger model like `e5-large-v2` or an OpenAI embedding API might better match paraphrased student questions (e.g., "how much is IV?" → International Village rates).
- **Context length** — some policy sections are dense; models with longer input windows reduce the need to split mid-rule.
- **Multilingual support** — less critical for this English-only corpus, but matters if expanding to international student FAQs.
- **Latency vs. local** — MiniLM is fast and free locally; hosted models add API cost and network delay but may improve retrieval on noisy review language.

---

## Grounded Generation

**System prompt grounding instruction:**

The LLM (`llama-3.3-70b-versatile` via Groq) receives a system message requiring context-only answers:

> You are a Northeastern University housing assistant. Answer ONLY using the Retrieved Documents below. Do not use outside knowledge.
>
> Rules:
> - If the documents do not contain enough information, respond exactly: "I don't have enough information on that in my documents."
> - Do not guess or infer beyond what the documents state.
> - When citing facts, mention the source filename in parentheses, e.g. (source: room_rates.txt).
> - For student reviews, distinguish them from official policy.

**Structural grounding choices:**

1. **Numbered context blocks** — retrieved chunks are formatted as `[1] (source: room_rates.txt | section: Kerr Hall (KER))` followed by chunk text, so the model can cite by filename.
2. **Distance filter** — chunks with cosine distance ≥ 0.65 are dropped before the LLM call; if none remain, the system returns the decline message without calling Groq.
3. **Low temperature** — `temperature=0.2` reduces creative hallucination.
4. **Programmatic source list** — the Gradio UI "Retrieved from" box is built from retrieval metadata (`sorted({c["source"] for c in chunks})`), not from LLM output alone.

**How source attribution is surfaced in the response:** The LLM is instructed to cite filenames in parentheses within the answer text (e.g., `(source: room_rates.txt)`). The UI also shows a bullet list of unique source filenames from the retrieval step.

---

## Evaluation Report

All 5 test questions from `planning.md` were run end-to-end through `query.ask()` on June 14, 2026.

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What is the per-semester rate for a standard double room at Kerr Hall for 2025–2026? | **$5,315** per semester (`room_rates.txt`) | Correctly stated $5,315 for Kerr Hall double standard; cited `room_rates.txt`. Top-5 chunks all from `room_rates.txt`, Kerr Hall ranked #1 (distance 0.26). | Relevant | Accurate |
| 2 | When is the housing application due for students entering in Fall 2026, and when must the enrollment deposit be paid? | Application due **May 7, 2026**; deposit by **May 1, 2026** | Correctly stated May 7, 2026 (application) and May 1, 2026 (deposit); cited `application_process.txt`. Also retrieved `spring_housing.txt` (NUin timeline, distance 0.38) which does not contain Fall 2026 entry dates. | Partially relevant | Accurate |
| 3 | What do RoomSurf students say about noise and wall thickness at International Village? | Reviews mention **thin walls**, floor/neighbor noise, **NUPD** nearby; some praise location and dining | Quoted 5 students on thin walls and noise; cited `dorm_review.txt`. Did **not** mention NUPD/police noise even though Kelsey Z.'s review (retrieved at rank 4) explicitly mentions "NUPD right next door." | Relevant | Partially accurate |
| 4 | On average, what housing styles are NUin spring returners placed into? | **85%** semi-private suites, **10%** apartment, **5%** traditional | Correctly stated 10% apartment, 85% semi-private suites, 5% traditional; cited `spring_housing.txt`. However, the **timeline/preference-form chunk ranked #1** (distance 0.45); the statistics paragraph ranked #2 (distance 0.50). Irrelevant `room_rates.txt` chunks also passed the distance filter (0.61). | Partially relevant | Accurate |
| 5 | Can students bring their own microwave or outside furniture to traditional or suite-style dorms? | **No** personal microwaves; rent micro-fridge instead. **No outside furniture** except desk chair; also prohibited: tapestry, AC units, halogen lamps, heating-coil appliances | Correctly stated no personal microwaves and no outside furniture (except desk chair); cited `what_to_bring.txt`. Top chunk was a 1,605-char block starting with "Medicine and Toiletries" — microwave/prohibited-item rules were at the **bottom** of the same chunk. Ranks 2–5 were off-target (`dorm_review.txt`, `room_rates.txt`, `university_housing.txt`). | Partially relevant | Accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:**

> On average, what housing styles are NUin spring returners placed into?

**What the system returned:**

The final answer was factually correct — "10% apartment style housing, 85% semi-private suites, and 5% traditional first-year halls" (source: `spring_housing.txt`). However, retrieval did not surface the best chunk first. The top-ranked chunk (distance 0.45) contained the NUin **timeline** (preference form availability, due dates) rather than the **housing statistics** paragraph. The statistics chunk ranked second (distance 0.50). Two irrelevant `room_rates.txt` chunks (distance 0.61) also passed the `MAX_DISTANCE = 0.65` filter and were sent to the LLM, adding noise.

**Root cause (tied to a specific pipeline stage):**

**Retrieval (embedding similarity ranking).** The query asks about "housing styles" and "placed into," which semantically overlaps with timeline language in `spring_housing.txt` ("Housing Application Preference Form," "Housing Preference Form Due"). The `all-MiniLM-L6-v2` embedding model ranked that timeline chunk higher than the statistics paragraph containing "Average placements: 10% apartment style housing, 85% semi-private suites, 5% traditional first-year halls." The percentages sit mid-document in a dense paragraph without a distinctive header, so the embedding signal for "85%/10%/5%" was weaker than the form-deadline phrasing. The distance filter (0.65) was too permissive, letting unrelated `room_rates.txt` chunks into the prompt.

**What you would change to fix it:**

1. **Chunking** — split `spring_housing.txt` so the Housing Statistics block (lines 6–8) is its own chunk with a clear header, rather than sharing a recursive split with timeline and LLC sections.
2. **Retrieval** — tighten `MAX_DISTANCE` to ~0.55 for this corpus, or use metadata filtering (e.g., boost chunks whose section contains "Statistics").
3. **Re-ranking** — add a lightweight keyword re-ranker that boosts chunks containing percentage patterns when the query asks about averages or placement rates.

---

## Spec Reflection

**One way the spec helped you during implementation:**

The `planning.md` chunking rules were specific enough to direct AI-generated code with minimal rework. The decision to use one chunk per building for `room_rates.txt` and one chunk per review for `dorm_review.txt` came directly from the Anticipated Challenges section — and spot-checking confirmed Kerr Hall rates stayed intact in a single chunk. The Architecture Mermaid diagram and Retrieval Approach (top-k = 5, MiniLM, ChromaDB) were pasted into AI prompts for Milestones 4 and 5, producing `vector_store.py` and `query.py` that matched the spec on the first pass.

**One way your implementation diverged from the spec, and why:**

The final chunk count was **144** rather than the estimated 80–120, because `room_rates.txt` alone produced 104 building-level chunks — more buildings than initially estimated. I also added a `MAX_DISTANCE = 0.65` pre-LLM filter that was not in the original spec; this prevents the model from answering when retrieval finds no reasonably similar chunks (e.g., out-of-domain dining hall questions). Finally, I pinned `huggingface-hub<1.0` in `requirements.txt` after Gradio 6.x pulled a newer version that conflicted with `sentence-transformers` — a dependency issue the spec did not anticipate.

---

## AI Usage

**Instance 1 — Milestone 3 ingestion and chunking**

- *What I gave the AI:* The Documents table and Chunking Strategy section from `planning.md`, plus `requirements.txt` constraints (no LangChain required, plain Python).
- *What it produced:* `models.py`, `chunking.py`, and `ingest.py` with recursive splitting (1600 chars / 240 overlap), special handlers for room rates and dorm reviews, and metadata tagging.
- *What I changed or overrode:* Removed footer-stripping logic that accidentally deleted most of `room_rates.txt` content (Contact Information block was too aggressive). Fixed loading of `Living_Learning_Communities.txt` (trailing space in filename) by switching from `glob` to `iterdir()`. Completed a truncated Midtown Hotel review in `dorm_review.txt` that was cut off mid-sentence in the source scrape.

**Instance 2 — Milestone 5 grounded generation and Gradio UI**

- *What I gave the AI:* The Evaluation Plan, Anticipated Challenges, Architecture stage 5 (Groq + grounded prompt), and the assignment Gradio skeleton structure.
- *What it produced:* `query.py` (strict system prompt, numbered context blocks, Groq `llama-3.3-70b-versatile`), `app.py` (Gradio Blocks UI), and `test_generation.py`.
- *What I changed or overrode:* Added `MAX_DISTANCE = 0.65` to drop weak retrieval matches before calling the LLM — the AI's first draft sent all top-5 chunks regardless of distance. Pinned `huggingface-hub>=0.34.0,<1.0` after Gradio installation broke `sentence-transformers` imports. Extended `test_generation.py` to cover all 5 evaluation questions (original draft only tested 3 in-domain + 1 out-of-domain).

---

## Demo Recording Checklist

Record a 3–5 minute demo video covering the items below. Use QuickTime, OBS, or similar screen capture.

1. **Launch the app** — run `python app.py`, open http://localhost:7860
2. **Three cited queries** (show answer text + "Retrieved from" sources):
   - Q1: "What is the per-semester rate for a standard double room at Kerr Hall for 2025-2026?"
   - Q3: "What do RoomSurf students say about noise and wall thickness at International Village?"
   - Q5: "Can students bring their own microwave or outside furniture to traditional or suite-style dorms?"
3. **Narrate one success** — e.g., Q1: retrieval returned Kerr Hall chunk #1, generation cited `room_rates.txt` with correct $5,315
4. **Narrate one failure/struggle** — run Q4: "On average, what housing styles are NUin spring returners placed into?" Explain that timeline chunk ranked above statistics chunk (see Failure Case Analysis)
5. **README walkthrough** — scroll through the Evaluation Report table and Failure Case Analysis sections

Upload the video per your course submission instructions.
