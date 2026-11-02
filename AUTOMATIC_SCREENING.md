# Automatic timing screening — 2 November 2026

Rule version: `timing-v1-20261102`. These are operational thresholds selected before recruitment, not empirically calibrated diagnostic cutoffs.

On each formal trial submission, the app evaluates the following rules:

| Rule | Automatic invalidity threshold |
| --- | --- |
| Stage 1 repeated timeout | At least 3 distinct trial records marked timeout; either rating page may exceed its 120-second limit |
| Stage 2 severe timeout | Both the 480-second search limit and 240-second answer/rating limit exceeded on the same puzzle |
| Stage 1 rapid submissions | 5 consecutive formal trials, each with prior and update pages totaling less than 5 seconds |
| Stage 2 rapid submissions | 3 consecutive formal puzzles, each with search and answer/rating totaling less than 10 seconds |

The triggering trial is retained. The app records a `screening_invalid` event with rule version, thresholds and reasons, sets `quality_status=invalid` and `payment_eligible=false`, then ends the task. It generates neither a completion code nor a manual-review code. Participant ID remains available for reporting technical problems. Local JSONL and Google Sheets logging remain enabled. Existing Sheet columns stay in place; new columns are appended.

Rules are checked during page interaction/submission, not by an unattended background countdown. Practice, individual query speed, correctness, Aha ratings, and fitted parameters do not trigger these rules. Missing timing data do not count as rapid submissions. A normal-duration or timeout trial breaks a rapid-submission streak. Other existing flags retain manual review. Recruitment-platform counting and payment processing remain with Credamo; the app itself does not change their survey quota or rejection accounting.

Participant-facing rules are stated on the formal-task introduction and in the Credamo consent. No recruitment messages were sent automatically.

Completed sessions use the same `INS-` completion-code format and participant-facing message for both `valid` and `review`. Backend `quality_status`, `quality_flags`, `quality_summary`, `payment_eligible`, and event type preserve the distinction. A completion code confirms submission, not quality approval. Previously issued `REVIEW-` codes remain valid lookup keys in historical data. Automatic-invalid sessions still receive no completion code.
