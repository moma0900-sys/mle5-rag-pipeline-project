# Evaluation Report — Northstar RAG Pipeline

Documentation and verification date: October 22, 2026.

## 1. Results and evaluation basis

The local pipeline answers 12 of the 13 answerable questions and correctly abstains on all 3 unanswerable questions. For `q09`, it incorrectly abstains despite retrieving the required sources. This yields 15 correct answer-or-abstain decisions across 16 cases, or 93.75% decision accuracy.

The evaluation is based on the saved run in [evaluation_results.json](evaluation_results.json). That file contains all 16 cases, their answers, original hits, selected evidence, metrics, and scoring rules. Definitions and aggregations were checked against the underlying data. Final labels were not changed.

The question set is now named [evaluation_questions.json](../data/evaluation/evaluation_questions.json) in the repository; it is the final set called `questions.json` in the assignment. Calibration uses the separate [calibration.json](../data/evaluation/calibration.json) file.

The final JSON file was read for this report rather than overwritten by a new final model run. Tests, corpus structure, and the six calibration cases were checked again. The JSON file does not contain a creation timestamp or complete configuration snapshot; the following configuration was verified against the current code.

## 2. Configuration and implementation choices

| Component | Verified configuration |
|---|---|
| Sources | 10 Markdown files, 22 chunks. |
| Chunking | Preserve paragraph boundaries; at most 100 words, with 15-word overlap only within long paragraphs. |
| Metadata | `chunk_id`, `doc_id`, `title`, `text`, `position`. |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2`. |
| Model revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. |
| Model execution | CPU; title and chunk text embedded together; questions use the same model. |
| Vectors | Normalized NumPy `float32` vectors with 384 dimensions. |
| Backend | Qdrant Local, `QdrantClient(":memory:")`. |
| Collection | `northstar_support`, cosine similarity, 384 dimensions. |
| Retrieval | At most 3 chunk hits; payloads returned, vectors omitted. |
| Filter order | Safety filter, then score checks, then answer generation. |
| Abstention threshold | The best remaining hit must score at least 0.45. |
| Evidence threshold | Individual passages require a positive score of at least 0.40. |
| Answer mode | Deterministic, verbatim extractive output with `[doc_id]`; no generative LLM. |

Paragraph-based splitting is intended to avoid breaking related policy statements unnecessarily. The maximum window size limits long paragraphs, while overlap preserves boundary context. The longest chunk in the current corpus contains only 43 words. Window size and overlap therefore do not affect these short paragraphs; they are starting values rather than a demonstrated optimum.

In-memory Qdrant is straightforward for 22 chunks and requires no service. Each run rebuilds the index. The trade-offs are repeated computation and lack of persistence. Usage instructions and implementation details are provided in [PROJECT_SOLUTION.md](../PROJECT_SOLUTION.md).

Versions installed during the documentation review: Python 3.13.15, Sentence Transformers 5.5.1, Qdrant Client 1.18.0, NumPy 2.5.4, Pydantic 2.13.4, PyTorch 2.14.1, and pytest 9.0.3. The shared `uv.lock` is the basis for environment setup.

## 3. Calibration of abstention rules

The thresholds were selected during development using the six calibration cases and fixed before final evaluation. During report preparation, the same configuration was reproduced offline using the cached model; thresholds were not optimized again.

| Case | Expected abstention | Best score | Selected passages | Actual abstention |
|---|---|---:|---:|---|
| cal01 — Duplicate-payment details | No | 0.549939 | 3 | No |
| cal02 — Lost authenticator | No | 0.495570 | 1 | No |
| cal03 — API rate limiting | No | 0.715489 | 2 | No |
| cal04 — Free subscription extension after an outage | Yes | 0.310063 | 0 | Yes |
| cal05 — Annual-subscription cancellation refund | Yes | 0.441274 | 0 | Yes |
| cal06 — Student discount | Yes | 0.293542 | 0 | Yes |

For these cases, the overall threshold of 0.45 separates the strongest unanswerable top hit (cal05: 0.441274) from the weakest answerable top hit (cal02: 0.495570).

Individual passages may still be retained from 0.40 upward. In cal01, the invoice number and payment references appear in `billing-refunds-0`, which scores 0.445552. Applying 0.45 to every passage would remove this important evidence even though the overall question passes the gate.

All six abstention decisions match their labels. This reflects calibration on a small set, not independent evidence of reliability. High cosine similarity does not establish that a retrieved policy applies to the request.

## 4. Scoring rules and metrics

All individual scores range from 0 to 1. Each aggregate is the arithmetic mean of the applicable individual scores. `null`, shown as “—” in tables, is excluded rather than counted as a success.

### Recall@3

For each answerable question:

```text
Recall@3 = Number of expected document IDs in the original top-3 hits
           / Number of expected document IDs
```

Document IDs are treated as sets for recall; multiple chunks from the same source do not count more than once. The metric covers 13 answerable questions. It evaluates retrieval before safety filtering and score checks, not the final evidence selection.

### MRR@3

For each answerable question, use the reciprocal rank of the first relevant chunk in the top-3 hits. A chunk is relevant when its document ID belongs to the expected sources. If none is relevant, the score is 0. The chunk ranking is not deduplicated by document.

Twelve questions have their first relevant hit at rank 1, and one at rank 2:

```text
MRR@3 = (12 × 1 + 1 × 1/2) / 13 = 0.961538…
```

### Answer relevance: deterministic term-coverage proxy

The JSON metric `answer_relevance` is the fraction of supplied `required_terms` found in the answer. Case is normalized using `casefold()`, and whitespace using `split()` and `join()`. The check uses substrings, not semantic analysis. An incorrect abstention on an answerable question receives 0. Unanswerable questions are excluded.

This evaluates 13 questions: twelve with complete term coverage and q09 with 0. The score does not measure conciseness or whether terms are used consistently and appropriately for the question. For example, a bare “30” can occur in an unrelated context. This metric is an explicitly disclosed proxy, not a comprehensive assessment of answer relevance.

### Faithfulness: literal support

Each returned answer is split into passages at blank lines. A passage passes if its complete text and final citation exactly match the rendered form of a selected evidence passage:

```text
<Evidence text> [<doc_id>]
```

The per-answer score is the fraction of passages that pass. The metric covers the 12 returned answers; abstention messages are excluded. During report verification, the selected evidence also matched the chunks generated from the current source documents.

This rule suits verbatim extractive output. It does not establish policy applicability and would be inappropriate for freely paraphrased LLM answers. Checking citation IDs alone would not be sufficient for this assessment.

### Decision accuracy

All 16 questions are included. A decision receives 1 when `actual_abstain == expected_abstain`, otherwise 0. The metric evaluates whether the system answers or abstains, not necessarily the correctness of an answer it returns.

| Metric | Value | Evaluated cases |
|---|---:|---:|
| Recall@3 | 1.000000 = 100% | 13 |
| MRR@3 | 0.961538 | 13 |
| Answer relevance, term-coverage proxy | 0.923077 = 92.31% | 13 |
| Faithfulness, literal support | 1.000000 = 100% | 12 |
| Decision accuracy | 0.937500 = 93.75% | 16 |

Answer-or-abstain decisions are distributed as follows:

| Expected outcome | Answer returned | Abstained |
|---|---:|---:|
| Answerable | 12 | 1 |
| Unanswerable | 0 | 3 |

## 5. Per-question results

R = Recall@3; RR = reciprocal rank of the first relevant hit; A = term-coverage proxy; F = literal support. Scores are exact 0/1 values or the stated reciprocal rank. Top scores are rounded to three decimal places for readability. Full questions and sources are available in the linked JSON file.

| ID | Topic | Top score | R | RR | A | F | Output | Decision correct |
|---|---|---:|---:|---:|---:|---:|---|---|
| q01 | Duplicate payment: details and deadline | 0.634 | 1 | 0.5 | 1 | 1 | Answer | Yes |
| q02 | Finance review time | 0.756 | 1 | 1 | 1 | 1 | Answer | Yes |
| q03 | Correcting a company name on an invoice | 0.762 | 1 | 1 | 1 | 1 | Answer | Yes |
| q04 | Lost authenticator | 0.623 | 1 | 1 | 1 | 1 | Answer | Yes |
| q05 | Password-reset link expiry | 0.821 | 1 | 1 | 1 | 1 | Answer | Yes |
| q06 | Details required for an assignment extension | 0.518 | 1 | 1 | 1 | 1 | Answer | Yes |
| q07 | Handling HTTP 429 | 0.519 | 1 | 1 | 1 | 1 | Answer | Yes |
| q08 | Frequency of incident updates | 0.680 | 1 | 1 | 1 | 1 | Answer | Yes |
| q09 | Authentication tokens in model prompts | 0.347 | 1 | 1 | 0 | — | Abstention | No |
| q10 | Instructions in retrieved documents | 0.620 | 1 | 1 | 1 | 1 | Answer | Yes |
| q11 | Cafeteria menu | 0.168 | — | — | — | — | Abstention | Yes |
| q12 | Annual enterprise-subscription price | 0.225 | — | — | — | — | Abstention | Yes |
| q13 | Passport scan in chat for an MFA reset | 0.557 | 1 | 1 | 1 | 1 | Answer | Yes |
| q14 | Immediate deadline change upon request | 0.771 | 1 | 1 | 1 | 1 | Answer | Yes |
| q15 | Reporting an unexpected absence | 0.590 | 1 | 1 | 1 | 1 | Answer | Yes |
| q16 | Reserving bicycle parking | 0.110 | — | — | — | — | Abstention | Yes |

## 6. Failures and fragile successes

### q09: Correct sources retrieved, but an incorrect abstention

The question is: “Can I send authentication tokens to a model while debugging support requests?”

The original hits are:

1. `technical-incidents-1`: 0.347422.
2. `privacy-data-0`: 0.327595.
3. `account-mfa-1`: 0.302647.

Both expected sources are retrieved. `privacy-data-0` contains the rule never to include authentication tokens, among other sensitive items, in a model prompt. Nevertheless, the best score is below 0.45. No evidence is selected, and the system abstains.

The cause is the score gate, not an absence of relevant sources in the top 3. Recall and RR are 1, but the abstention decision is incorrect. This matters in support because an important privacy rule is not returned. The run remains documented without changes; tuning the threshold using q09 would turn the already inspected final set into development data.

### q01: Correct details, but imprecise ranking and excessive context

For the question about duplicate-payment details and deadlines, `invoice-corrections-1` ranks first at 0.634044. It is followed by `billing-refunds-2` at 0.505183 and the necessary `billing-refunds-0` at 0.470182.

The output contains the invoice number, both payment references, and the 30-calendar-day deadline, but first includes additional rules about invoice corrections and other refund requests. All expected terms are present even though the answer is unnecessarily long. This case illustrates the limitation of the term-coverage proxy and motivates more precise context selection.

### q13: Correct central message, but only one expected source is used

The original search retrieves `account-mfa` and `privacy-data`; Recall@3 is 1. After filtering, only `account-mfa-1` remains, scoring 0.557130. `privacy-data-1`, at 0.320020, falls below the evidence threshold.

The remaining paragraph states that support never asks for passport scans or identity documents in chat and covers the required terms. The answer therefore succeeds under the defined rubric. The case also demonstrates that complete retrieval recall does not mean all expected sources are incorporated into the answer.

### q10: Safety rule answered, but the final case does not exercise the attack

The top-3 hits are `retrieved-content-safety-0`, `retrieved-content-safety-2`, and `privacy-data-0`. The attack-example chunk, `retrieved-content-safety-1`, is not retrieved.

The answer contains the correct rules. However, this evaluation case does not demonstrate that an actually retrieved attack was blocked. The separate synthetic safety tests provide targeted, limited evidence for that behavior.

## 7. Safety behavior and tests

The application treats retrieved text as data. Its current extractive output does not interpret text as executable instructions. In addition, `railguards.py` excludes chunks containing known suspicious phrases. The warning identifies the source and chunk. Only afterward does the application check whether the remaining hits meet the score thresholds.

The filter is a heuristic. Rewordings, other languages, encodings, or other attack objectives may go undetected; harmless descriptions may also be excluded. Warnings are console output and are not currently persisted in the JSON file.

All seven custom tests passed on October 22, 2026. They cover:

1. Removing a suspicious chunk while retaining an unflagged hit.
2. Abstaining when removal leaves only a hit scoring 0.42.
3. Returning an accepted passage with a citation.
4. Rejecting a missing citation.
5. Rejecting a fabricated source.
6. Chunk text, overlap, document association, and position counting.
7. Qdrant search using synthetic vectors and their complete associated payloads.

These tests are not a comprehensive security assessment or a replacement for semantic evaluation. Metric calculations were additionally checked in memory using the saved data and synthetic counterexamples; there is currently no separate persisted pytest test for those calculations.

## 8. Three prioritized improvements

### Priority 1: Assess applicability and abstention more reliably

Build a larger independent calibration set with nearby but unsupported requests and privacy-related formulations. Beyond cosine scores, assess support from specific passages, for example through hybrid retrieval followed by reranking. Evaluate new rules on a new, previously unused test set.

The goal is to reduce false abstentions such as q09 without authorizing unsupported requests solely because of high similarity. Do not directly retune the threshold on the already inspected final set.

### Priority 2: Improve answer relevance and readability

Select the sentences that actually answer the question within retrieved sources, reduce redundant or merely related rules, and preserve evidence attribution. Extend evaluation with a documented manual relevance assessment covering completeness, contradictions, and unnecessary content.

q01 provides a concrete starting point. This improvement does not require generative LLM integration.

### Priority 3: Extend safety and evaluation evidence

Add adversarial tests, multilingual and obfuscated attacks, harmless counterexamples, and source-manipulation cases. Persist structured exclusion logs without recording secrets or complete attack contents. Save timestamps, configuration, and source versions with results, and test metric calculations automatically.

Before use by support staff, also introduce a controlled pilot with human approval and clear escalation. The existing phrase list is not a production-grade defense.

## 9. Answers to the four submission questions

### 1. Which retrieval failures matter most in support, and why?

Incorrectly accepted, thematically similar policies can produce false commitments about refunds, deadlines, or privacy. The opposite error also matters: in q09, the required sources are retrieved, but the abstention rule suppresses the safety guidance. Evaluation and diagnosis must therefore distinguish retrieval, evidence selection, and actual output.

### 2. How was the no-answer threshold selected?

Using the six calibration cases: 0.45 separates the observed top scores of answerable and unanswerable questions. The additional evidence threshold of 0.40 retains supplementary passages for an accepted question, particularly the important cal01 chunk scoring 0.445552. These values were fixed before final evaluation and reproduced unchanged during report preparation.

### 3. What evidence supports faithfulness to the sources?

Answers consist of verbatim evidence passages with source IDs. Automated citation validation rejects missing or foreign IDs. Evaluation additionally checks the entire passage together with its citation; all twelve returned answers satisfy this rule. Source verification for the report confirms their origin in the current documents. This demonstrates literal support, not automatic applicability or optimal answer relevance.

### 4. What must change before use in real support?

A larger independent evaluation set, policy-applicability checks, more precise output, stronger security tests, and traceable execution logs are needed. Actual privacy requirements, document maintenance, access rights, and responsibilities must be established in the operating environment. A supervised pilot with human review is an appropriate starting point; this small teaching dataset does not justify autonomous production deployment.

## 10. Reproduction and verification limits

From the repository root:

```powershell
uv sync --locked --group dev
uv run pytest tests -q
uv run main.py
```

The first command is the documented setup procedure using the lockfile. A completely fresh installation was not performed during report preparation. The seven tests and offline calibration succeeded in the existing environment. The last command rebuilds the in-memory index, reruns final evaluation, and overwrites the JSON file; it does not update this Markdown report.

Keep the settings fixed for reproduction. Changes to sources, questions, or methods require treating the run and report as a new experiment. The small teaching corpus, 13 answerable and 3 unanswerable final cases, and six calibration cases substantially limit the conclusions.

