# Human Experiment Platform

This Streamlit app is the core human experiment platform. Credamo, Wenjuanxing,
or another survey platform should be used as the outer recruitment/payment
layer only.

## Recommended Online Flow

```text
survey platform screening / consent / demographic quota
  -> Streamlit URL with participant parameters
  -> Stage 1 passive update + Stage 2 active query task
  -> Streamlit completion code
  -> participant returns to survey platform
  -> completion-code validation and payment
```

Do not replace the core task with fixed survey questions unless the theoretical
Stage 2 active-query component is intentionally removed.

## URL Parameters

The app can receive participant and recruitment metadata from the URL.

Example:

```text
https://your-app.streamlit.app/?pid=${participant_id}&source=credamo&group=FH&condition=balanced_stage1
```

Supported aliases:

```text
participant id: pid, participant_id, participantId, uid, id
platform id:    platform_pid, external_id, respondent_id, wjx_id, credamo_id
source:         source, platform, recruitment_source
group:          group
condition:      condition, cond
return link:    return_url, redirect
```

If `return_url` contains `{pid}` or `{completion_code}`, the app replaces those
placeholders at the end of the experiment. Otherwise it appends `pid` and
`completion_code` as query parameters.

## Streamlit Secrets

Use the balanced Stage 1 materials by default:

```toml
materials_file = "materials_for_app_balanced_stage1.json"
stage1_lookup_prefix = "lookup_stage1_balanced"
full_lookup_prefix = "lookup_full"
condition = "balanced_stage1"

stage1_n = 30
stage2_n = 5
min_queries = 3
max_queries = 8

require_desktop = true
instruction_video_url = ""

stage1_prior_timeout_sec = 180
stage1_update_timeout_sec = 180
stage2_query_timeout_sec = 480
stage2_answer_timeout_sec = 240

worksheet_name = "human_events_v2"
gsheet_url = "https://docs.google.com/spreadsheets/d/..."
randomization_salt = "replace-with-private-randomization-salt"
completion_salt = "replace-with-private-completion-salt"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
```

For a shorter human session, reduce `stage1_n` and/or `stage2_n` only after the
final IRB design is decided. The material file can stay the same.

## Required Runtime Files

The app expects these files under `experiment_v2_edit/data/`:

```text
materials_for_app_balanced_stage1.json
lookup_stage1_balanced_FH.json
lookup_stage1_balanced_FN.json
lookup_stage1_balanced_MH.json
lookup_stage1_balanced_MN.json
lookup_full_FH.json
lookup_full_FN.json
lookup_full_MH.json
lookup_full_MN.json
query_grounding_aliases.json
semantic_grounding_index.npz
```

The full hyperbolic embedding is not required at runtime. The app uses
precomputed lookup files plus the compact semantic grounding index for query
resolution.

## Data Recording

The app writes every event locally to:

```text
experiment_v2_edit/logs/human_events.jsonl
```

If Google Sheet append fails after retries, the failed row is also saved to:

```text
experiment_v2_edit/logs/human_events_unsynced.jsonl
```

Main event types:

```text
stage1_passive_update
stage2_active_query
stage2_final_guess
completion
```

Important quality-control fields include:

```text
participant_id
platform_participant_id
platform_source
condition
desktop_confirmed
session_id
elapsed_on_screen_sec
screen_timeout_sec
timeout_flag
invalid_reason
match_type
match_confidence
query_history
known_story
aha
aha_suddenness
aha_surprise
completion_code
completion_code_hash
```

## Participant-Facing Flow

The current formal flow is:

```text
intro / consent / desktop confirmation
  -> instruction page with optional video
  -> practice Stage 1
  -> practice Stage 2
  -> formal Stage 1 passive update
  -> formal Stage 2 query page
  -> formal Stage 2 answer + aha questionnaire page
  -> completion code
```

Stage 2 only shows participants their own query words and the 0-100 feedback
scores. Resolver details such as matched word, match type, semantic candidates,
and raw probabilities are stored in the event log but are not shown on screen.

The formal study should remain desktop/laptop-only unless the IRB and analysis
plan explicitly allow mobile participation. Mobile use changes reading,
typing, and query behaviour, so non-desktop sessions should be treated as pilot
or quality-control exceptions.

## Public Repo Warning

If the deployed Streamlit repo is public, materials and lookup files can expose
answers or high-scoring query words. For formal data collection, prefer a
private deployment or a deployment method where data files are not visible in a
public GitHub repository.

If a public Streamlit deployment is unavoidable, use the survey platform to
control access, pass unique participant IDs, verify completion codes, and screen
for repeated or suspicious submissions.

## Local Test

```bash
cd /Users/qujianhui/Documents/Codex/2026-06-17/overleaf-tex-root-thuthesis-example-tex
streamlit run experiment_v2_edit/human_app.py
```

Test with URL parameters:

```text
http://localhost:8501/?pid=test_FH_001&source=local&group=FH&condition=balanced_stage1
```
