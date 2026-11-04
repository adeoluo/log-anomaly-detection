# Threat Model - Log Anomaly Detection

This is a defensive monitoring tool: it reads logs and metrics and raises alerts. It executes no
code from its inputs and calls no external services.

## Assets

- **Log content.** Production logs can hold hostnames, IP addresses, user names, paths and
  occasionally secrets. The parser keeps templates (with variables masked) and counts, and the
  service returns templates and alert summaries, not raw lines.
- **Alert integrity.** Operators act on alerts; suppressed alerts hide incidents and spurious
  alerts cause fatigue.
- **Service availability** of the HTTP endpoints used by collectors.

## Trust boundaries

1. Log lines and metric values arriving through the CLI (`stream`, `regress`) or the HTTP service
   (`/parse`, `/ingest`, `/score`, `/regress`) are untrusted.
2. Training logs chosen by the operator are trusted to be mostly normal.
3. Datasets downloaded by `scripts/download_data.sh` come from Zenodo and GitHub over HTTPS.

## Threats and mitigations

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Resource exhaustion | Huge request bodies or line batches | Memory or CPU exhaustion | Request bodies capped (`LAD_MAX_BODY_BYTES`, 4 MB default), at most 100,000 lines per request, files streamed in chunks |
| Template explosion | Many unique messages (random tokens) | Unbounded parse tree growth, slow matching | Masking regexes collapse variables; growth is visible in the `lad_templates` gauge; operators can restart with a fresh parser |
| Regex denial of service | Pathological input against masking regexes | CPU burn | The Rust core uses the `regex` crate (linear time, no backtracking); the Python reference is not used in the service by default |
| Training-data poisoning | Anomalies in the training window | Detector learns failures as normal | `--labels` drops known-anomalous sessions; the evaluation protocol trains on normal data only; retraining is explicit |
| Evasion | Attacker crafts logs that mimic normal templates | Missed alert | Out of scope for template-count detectors; documented as a limitation; combine with other signals |
| Sensitive data in alerts | Raw lines echoed back | Data exposure | Alerts carry masked templates, keys and scores, not raw lines |
| Unauthenticated access | Service exposed on a network | Anyone can ingest or read templates | Binds to 127.0.0.1 by default; put it behind an authenticating proxy if exposed |
| Supply chain | Malicious dependency | Code execution | Dependencies pinned in `uv.lock`; CodeQL in CI; few runtime dependencies (numpy, pydantic) |

## Residual risk

Template-count and next-event detectors only see what the parser extracts; an adversary who
controls log content can shape it. The service has no authentication of its own. Both are
acceptable for an internal monitoring component and are stated in the README limitations.
