# RAG Baseline

First end-to-end run of the 7 questions in [`rag-checks.md`](rag-checks.md), recorded before any tuning.
This file is an evaluation record, not corpus input: the loader only reads `data/documents/*.txt`.

## Settings

| Item | Value |
|------|-------|
| Run date | 2026-10-30 |
| Code | `18e103c` + uncommitted `rag/retriever.py`, `rag/generator.py`, `scripts/ask.py` |
| Corpus | 3 documents (`plans.txt`, `billing.txt`, `account.txt`), 15 chunks |
| Chunking | one chunk per section; split by sentence only above 300 tokens (`max_tokens=300`, `overlap_tokens=50`); each chunk prefixed with `<title> > <section>`; tiktoken `cl100k_base` |
| Embedding | `text-embedding-3-small`, 1536 dimensions |
| Retrieval | pgvector cosine distance (`<=>`), `k=3`, filtered to the current embedding model |
| Generation | `gpt-6-luna`, reasoning effort `low`, `max_output_tokens=1000`, strict JSON schema `{answerable, answer, citations}` |
| Runs | `scripts/ask.py` once per question, no retries |

Accuracy is judged on whether numbers, conditions, and exceptions match the documents, not on wording.
Format and citation-label checks in the code do not establish accuracy; the judgments below are a manual review.

## Results

| # | Question | answerable | Answer (actual) | Cited (file / section) | Accuracy vs. expected | Missing conditions / unsupported claims |
|---|----------|-----------|-----------------|------------------------|-----------------------|----------------------------------------|
| 1 | How much storage does the Free plan include, and what happens when I reach it? | true | 1 GB of total storage [S1]. At the limit new uploads are blocked; existing files stay viewable, editable, downloadable [S1]. Quillbox does not delete files because you reach a plan limit [S2]. | S1 `plans.txt` / Free Plan; S2 `plans.txt` / What Happens When You Exceed a Limit | Correct: 1 GB, uploads blocked, existing files usable. | None missing. Minor wording drift: S2 says files are not deleted when you *exceed* a limit; the answer says *reach*. Not wrong here, but it is a paraphrase, not a quote. |
| 2 | What is the daily API request limit on the Team plan? | true | 50,000 API requests per day per workspace [S1]. | S1 `plans.txt` / Team Plan | Correct: 50,000 per day per workspace. | Omits what happens over the limit (HTTP 429 until 00:00 UTC), which `rag-checks.md` expects. The supporting chunk was retrieved as S3 but not used. |
| 3 | Can I get a refund for a monthly renewal charge? | true | Monthly renewals are not refundable [S1]. Duplicate charges or billing errors caused by Quillbox are refunded in full regardless of when reported [S1]. | S1 `billing.txt` / Refund Policy | Correct, including the billing-error exception. | Does not say that only first-time purchases and annual renewals have the 14-day refund window (expected answer mentions it). Not required to answer "no", but useful context. |
| 4 | If I cancel my Pro subscription, when do I lose Pro features? | true | Pro features stay active until the end of the current billing period. Then the workspace moves to Free [S1]. | S1 `billing.txt` / Canceling a Subscription | Correct timing and Free transition. | Omits that the account and data are kept (expected answer). The first sentence has no citation; the code only checks that labels match, not that every factual sentence is cited. |
| 5 | How long is a password reset link valid? | true | 60 minutes after it is sent [S1]. | S1 `account.txt` / Resetting Your Password | Correct: 60 minutes. | Omits that each link is single-use (expected answer). |
| 6 | Can I recover my account after requesting deletion? | true | Yes, within 30 days: sign in and select "Restore account"; documents and files come back, but the account returns on the Free plan because the paid subscription was already canceled [S1]. After 30 days it is permanently deleted [S1]. | S1 `account.txt` / Deletion Grace Period | Correct and complete: 30 days, "Restore account", Free plan on restore, permanent after 30 days. | None. |
| 7 | Does Quillbox offer a student or nonprofit discount? | false | The provided documents do not contain information about student or nonprofit discounts. | None (S1–S3 retrieved, none cited) | Correct: reports missing information; does not claim discounts are not offered; does not treat "two months free" on annual billing as a discount. | None. |

## Summary

- Retrieval: the expected source was rank 1 for all 6 answerable questions.
- Accuracy: 6/6 answerable questions have no wrong numbers, conditions, or exceptions; question 7 correctly reports that the documents have no answer.
- Unsupported claims: none found.
- Main weakness: answers are minimal. Secondary facts that `rag-checks.md` expects (Q2 HTTP 429 / reset time, Q4 data kept, Q5 single-use link) are left out even when the supporting chunk was retrieved.
- Citation check gap: Q4 shows a factual sentence without a citation, which passes the current validation.
- Reasoning tokens were 0 for every call at effort `low`.

## Usage

| # | Input tokens | Output tokens | Reasoning tokens | Total |
|---|-------------|---------------|------------------|-------|
| 1 | 797 | 89 | 0 | 886 |
| 2 | 784 | 44 | 0 | 828 |
| 3 | 723 | 63 | 0 | 786 |
| 4 | 746 | 57 | 0 | 803 |
| 5 | 780 | 44 | 0 | 824 |
| 6 | 817 | 102 | 0 | 919 |
| 7 | 700 | 34 | 0 | 734 |
| Total | 5,347 | 433 | 0 | 5,780 |

Generation cost at the published standard rate ($0.10 input / $0.50 output per 1M tokens): about $0.0008 for all 7 questions. Question embeddings are extra and much smaller.

## Revision 1 — prompt change (2026-10-30)

Goal: include only conditions and exceptions that change the answer, cite every factual sentence or bullet, and stop listing unrelated policies.
Not a goal: adding every secondary fact from `rag-checks.md` (for example single-use links or HTTP 429 details).

Changes in `rag/generator.py` `INSTRUCTIONS` only (code, schema, retrieval, chunking, model, effort, `k` unchanged):

| Rule | Before | After |
|------|--------|-------|
| 2 | After each sentence that states a fact, cite its source label in square brackets, e.g. [S1]. | End every sentence or bullet that states a fact with its source label in square brackets, e.g. [S1], even when consecutive sentences come from the same source. |
| 6 | Be concise, but include conditions and exceptions from the sources that affect the answer. | Answer the question directly. Include a condition or exception only when it changes the answer to this question; do not list policies the question does not ask about. |

Tests: 37/37 pass without API calls.

### Re-run: question 4 only, once

| Field | Result |
|-------|--------|
| answerable | true |
| Answer | "Your Pro features remain active until the end of your current billing period. When that period ends, your workspace moves to the Free plan automatically. [S1]" |
| Cited | S1 `billing.txt` / Canceling a Subscription |
| Accuracy | Correct: features end at the end of the current billing period, then Free. No unrelated policies listed. |
| Citation per sentence | **Not improved.** The first sentence still has no citation; one label covers both sentences. |
| Usage | input 775, output 57, reasoning 0, total 832 |

Conclusion: in this single run the prompt change alone did not fix the uncited sentence.
One run is not enough to judge the rate in either direction; the current code check (labels match) still cannot detect this case.

## Revision 2 — paragraph-level citations allowed (2026-10-30)

Decision: consecutive sentences based on the same source may share one citation at the end of the paragraph or bullet.
A new citation is still required whenever the source changes, and every cited label must be one of the supplied sources.

Reason:
- Revision 1's single re-run of question 4 produced exactly this pattern (two sentences from S1, one `[S1]` at the end). The answer was accurate and the one citation pointed to the source of both sentences.
- For short answers, repeating the same label after every sentence adds noise without making the source clearer.
- Enforcing per-sentence citations reliably would need a code check and a policy for rejected answers (error or retry); that is out of scope for this version.

Change in `rag/generator.py` `INSTRUCTIONS` rule 2 only:

| Before (Revision 1) | After |
|---------------------|-------|
| End every sentence or bullet that states a fact with its source label in square brackets, e.g. [S1], even when consecutive sentences come from the same source. | Cite the source label in square brackets, e.g. [S1], for every factual statement. Consecutive sentences based on the same source may share one citation at the end of the paragraph or bullet; start a new citation whenever the source changes. |

Unchanged: rule 6 from Revision 1, the citation-label validation in code (labels must be supplied sources, inline labels must match the `citations` list, answerable answers must cite something), model, effort, `k`, retrieval, chunking.
Not added: per-sentence citation checks, automatic retries.

Tests: 37/37 pass without API calls. No live API run for this revision.

Known limitation: the code still cannot tell whether a paragraph-final citation actually covers every sentence before it, or whether a paragraph mixes sources under one label. That remains a manual review item.
