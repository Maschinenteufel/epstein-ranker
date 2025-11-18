# Epstein Ranker

LLM-powered tooling for triaging the **U.S. House Oversight Epstein Estate** document release.  
This project:

1. Streams the document corpus through a locally hosted, open-source model (`openai/gpt-oss-120b` running via **LM Studio**) to produce ranked, structured leads.
2. Ships a dashboard (`viewer/`) so investigators can filter, chart, and inspect every scored document (including the full source text) offline.

The entire workflow operates on a single MacBook Pro (M3 Max, 128 GB RAM). With an average draw of ~100 W, a 60-hour pass consumes ≈6 kWh (~$1.50 at SoCal off-peak rates) with zero cloud/API spend.

---

## Data source & provenance

The repository’s base dataset is the **“20,000 Epstein Files”** text corpus prepared by [**tensonaut**](https://www.reddit.com/r/LocalLLaMA/comments/1ozu5v4/20000_epstein_files_in_a_single_text_file/), who OCR’d ~25,000 pages released by the U.S. House Committee on Oversight and Government Reform.  
Key references:

- Hugging Face dataset: <https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K>  
- Original release: [Oversight Committee Releases Additional Epstein Estate Documents (Nov 12, 2025)](https://oversight.house.gov/release/oversight-committee-releases-additional-epstein-estate-documents/)

**Corpus outline (summarized from the dataset card):**

- >25,000 plain-text documents derived from the committee’s Google Drive distribution.
- `TEXT/` files were converted directly from native text sources; `IMAGES/` files (≈20k JPGs) were OCR’d with Tesseract.
- Filenames preserve the source path so you can cross-reference with the official release.
- No manual edits beyond text extraction/OCR; expect OCR noise, redaction markers, or broken formatting.
- Legal status: documents originate from the House release and retain any original copyright. This repo asserts no ownership and offers no legal advice—ensure your use complies with applicable law.
- Content warning: expect references to sexual abuse, exploitation, trafficking, violence, and unverified allegations.

Huge thanks to **tensonaut** for the foundational OCR and dataset packaging; this project simply layers ranking and analytics on top.

---

## Requirements

- Python 3.9+
- `requests`
- LM Studio (or another OpenAI-compatible gateway) serving `openai/gpt-oss-120b` locally at `http://localhost:5002/v1`
- The dataset CSV (`data/EPS_FILES_20K_NOV2026.csv`). **Not included in this repo**—download it from the Hugging Face link above and place it in `data/` (see `data/README.md` for instructions).

Install Python deps (only `requests` is needed):

```bash
python -m pip install -r requirements.txt  # or just: python -m pip install requests
```

---

## Running the ranker

```bash
cp ranker_config.example.toml ranker_config.toml  # optional defaults
python gpt_ranker.py \
  --input data/EPS_FILES_20K_NOV2026.csv \
  --output data/epstein_ranked.csv \
  --json-output data/epstein_ranked.jsonl \
  --endpoint http://localhost:5002/v1 \
  --model openai/gpt-oss-120b \
  --resume \
  --sleep 0.5 \
  --config ranker_config.toml \
  --reasoning-effort low
```

Notable flags:

- `--resume`: skips rows already present in the JSONL/checkpoint so you can stop/restart long runs.
- `--checkpoint data/.epstein_checkpoint`: stores processed filenames to guard against duplication.
- `--reasoning-effort low/high`: trade accuracy for speed if your model exposes the reasoning control knob.
- `--reasoning-effort low/high`: trade accuracy for speed if your model exposes the reasoning control knob.
- `--include-action-items`: opt-in if you want the LLM to list action items (off by default for brevity).
- `--max-rows N`: smoke-test on a small subset.
- `--list-models`: query your endpoint for available model IDs.
- `--start-row`, `--end-row`: process only a slice of the dataset (ideal for collaborative chunking).
- `--overwrite-output`: explicitly allow truncating existing CSV/JSONL files (default is to refuse and require `--resume` or a different path).
- `--power-watts`, `--electric-rate`, `--run-hours`: plug in your local power draw/cost to estimate total electricity usage (also configurable via `ranker_config.toml`).
- `--power-watts`, `--electric-rate`, `--run-hours`: plug in your local power draw/cost to estimate total electricity usage (also configurable via the TOML file).

Outputs:

- `data/epstein_ranked.csv` – Spreadsheet-friendly table (headline, score, tags, power mentions, agencies, lead types).
- `data/epstein_ranked.jsonl` – Full JSON per document, including the original CSV row text for reproducibility.

---

## Scoring methodology (LLM prompt)

The model receives every row with the following instruction (excerpt):

```
You analyze primary documents related to court and investigative filings.
Focus on whether the passage offers potential leads—even if unverified—that connect influential actors ... to controversial actions, financial flows, or possible misconduct.
Score each passage on:
  1. Investigative usefulness
  2. Controversy / sensitivity
  3. Novelty
  4. Power linkage
Assign an importance_score from 0 (no meaningful lead) to 100 (blockbuster lead linking powerful actors to fresh controversy). Reserve 70+ for claims that, if true, would represent major revelations or next-step investigations.
Return strict JSON with fields: headline, importance_score, reason, key_insights, tags, power_mentions, agency_involvement, lead_types.
```

Rows ≥70 typically surface multi-factor leads (named actors + money trail + novelty). Anything below ~30 is often speculation or previously reported context.

---

## Interactive viewer

Serve the dashboard to explore results, filter, and inspect the full source text of each document:

```bash
./viewer.sh 9000
# or manually:
cd viewer && python -m http.server 9000
```

Open <http://localhost:9000>. Features:

- AG Grid table sorted by importance score (click a row to expand the detail drawer and read the entire document text).
- Filters for score threshold, lead types, power mentions, ad hoc search, and row limits.
- Charts showing lead-type distribution, score histogram, top power mentions, and top agencies.
- Methodology accordion describing the scoring criteria, prompt, and compute footprint.

`viewer/app.js` pulls from `data/epstein_ranked.jsonl`, so keep that file in sync with the latest run.

### Screenshots

| Table View | Insights & Charts |
| ---------- | ----------------- |
| ![Table view](imgs/table.png) | ![Insights + charts](imgs/graphs.png) |

| Methodology Explainer |
| --------------------- |
| ![Methodology explainer](imgs/info.png) |

---

## Collaborative ranking workflow

Want to help process more of the corpus? Fork the repo, claim a range of rows, and submit your results:

1. **Pick a chunk** – e.g., rows `00001–01000`, `01001–02000`, etc. Use whatever increments work. Announce the chunk (issue/Discord) so others don’t duplicate effort.
2. **Run the ranker on that slice** using the new range flags:

   ```bash
   python gpt_ranker.py \
     --config ranker_config.toml \
     --start-row 1001 \
     --end-row 2000 \
     --output contrib/epstein_ranked_1001_2000.csv \
     --json-output contrib/epstein_ranked_1001_2000.jsonl \
     --checkpoint contrib/.checkpoint_1001_2000.txt \
     --known-json data/epstein_ranked.jsonl \
     --resume
   ```

   This only processes documents in that range. `--known-json` makes the script aware of previously merged results (so duplicates are skipped automatically). Combine with `--resume` if you need to pause and continue later.

3. **Export your outputs** – store the resulting CSV/JSONL subset in `contrib/`. Suggested naming:

   - `contrib/epstein_ranked_<start>_<end>.csv`
   - `contrib/epstein_ranked_<start>_<end>.jsonl`

4. **Submit a PR** with your chunk. We’ll merge the contributions into the global dataset and credit collaborators in the README.

Guidelines:

- Do **not** commit the original 100 MB source CSV; each contributor should download it separately.
- Keep the JSONL chunks intact (no reformatting) so we can merge them programmatically.
- If you discover inconsistencies or interesting leads, open an issue to coordinate follow-up analysis.
- Pull the latest `data/epstein_ranked.jsonl` before starting; pass it via `--known-json` so you never duplicate work.

---

## Ethical considerations & intended use

- The corpus contains sensitive content (sexual abuse, trafficking, violence, unverified allegations). Use with care.
- Documents are part of the public record but may still be subject to copyright/privacy restrictions; verify before sharing or redistributing.
- Recommended use cases: investigative triage, exploratory data analysis, RAG/IR experiments, or academic review.
- This project does **not** assert any claims about the veracity of individual documents—scores merely prioritize leads for deeper human review.

---

## Acknowledgements

- **tensonaut** for compiling the OCR corpus and publishing it to Hugging Face.
- U.S. House Committee on Oversight and Government Reform for releasing the source documents.
- The LM Studio community & `r/LocalLLaMA` for pushing local LLM workflows forward.

---

## License

Released under the [MIT License](LICENSE). Please retain attribution to this project, the `tensonaut` dataset, and the U.S. House Oversight Committee release when building derivative tools or analyses.
