# Streamlit competition MVP audit

Audit date: 2026-11-17.

## Previous UI

The original `app.py` proved that upload → analysis → JSON download worked, but it
was a technical debug surface rather than a competition-ready product:

- the score was one of four plain metrics and lacked a strong visual hierarchy;
- component scores were an unlabeled generic chart;
- geometry reliability and gates were shown as raw JSON/dataframes;
- problem intervals had internal identifiers but no plain-language impact or fix;
- no representative video frames could be inspected;
- recommendations did not connect what/where/why/action in one place;
- the app returned early without an upload, hiding evaluation/method content;
- v1/v2/human evidence had no dedicated view;
- Streamlit reruns could repeat an expensive analysis of the same upload;
- the optional Streamlit runtime had not been exercised locally.

## Implemented MVP experience

- Branded responsive header and neutral competition-ready visual system.
- Prominent circular 0–100 score, suitability band, v1 delta, and explicit
  “heuristic, not probability” language.
- Five named score cards with weights and short reconstruction-oriented explanations.
- Evidence-level banner with all confidence/uncertainty notes.
- Filterable timestamped timeline with normalized full-video positions.
- Representative problem frames selected by deterministic issue priority. Blur and
  exposure are visible directly; motion/overlap captions explicitly state that the
  label describes the surrounding pair.
- Structured recommendation cards: what is wrong, exact affected time ranges, why
  reconstruction suffers, and how to record differently.
- Technical view retains gates, warnings, complete JSON, and JSON download.
- Evaluation view joins preserved v1/v2 evidence with human review. Three missing
  numerical scores stay “Not provided”; no value is invented.
- Method view explains scope, expected capture practice, and the separation from
  COLMAP validation.
- Upload SHA-256 session caching avoids re-analysis on ordinary Streamlit widget
  reruns. Temporary upload files are always deleted after analysis/frame extraction.

## UI testing

Pure presentation functions are tested without Streamlit: report components,
timeline enrichment, recommendations, evaluation joins, representative-frame
decoding, and source checksum preservation. With the optional UI dependency
installed, `streamlit.testing.v1.AppTest` renders the complete initial application,
verifies all three top-level views and the uploader, and reports no exceptions.

A real headless server smoke test used Streamlit 1.50.0. The first sandboxed bind
failed with `PermissionError: [Errno 1]`; the approved localhost-only retry started
at `127.0.0.1:8503`, returned `ok` from `/_stcore/health`, served the application
HTML, and shut down cleanly.

## Honest limitations

- Representative frames are sampled evidence, not an exhaustive scan of every
  source frame.
- A still image cannot visually prove shake, speed, overlap, or parallax; those
  labels apply to measured adjacent-frame intervals.
- The app does not segment the target object, so background features may influence
  geometry.
- Streamlit's configured upload-size limit still applies. The accessible 114.7 MB
  4K MOV is below Streamlit's default 200 MB limit.
- The four human-reviewed videos are audit cases, not an independent accuracy set.
- No sparse or dense reconstruction is run by the app.

