# Moshi FullDuplexBench evaluation

Self-contained duplex eval for base Moshi (Moshiko / Moshika). Scores the five
[FullDuplexBench](https://arxiv.org/abs/2503.04721) categories:

| Category | Dataset folder | Metrics |
| --- | --- | --- |
| Pause (Synthetic) | `synthetic_pause_handling` | take-over (`TO`) |
| Pause (Candor) | `candor_pause_handling` | `TO` |
| Backchannel | `icc_backchannel` | `TO`, `js_divergence`, `freq` |
| Smooth Turn Taking | `candor_turn_taking` | `TO`, `latency` |
| User Interruption | `synthetic_user_interruption` | `TO`, `latency`, LLM `score` 0–5 |

Stock Moshi has **no text system-prompt conditioner**, so PersonaPlex-style role
prompts are not applied. Generation is user wav in → Moshi audio + inner-monologue
text out.

The FullDuplexBench audio/metadata dump is **not** shipped in this repo. Obtain it
from the FullDuplexBench release and lay it out as:

```
bench-root/
  synthetic_pause_handling/     *.wav
  candor_pause_handling/        *.wav
  icc_backchannel/              *.wav
  candor_turn_taking/           *.wav + metadata *.json
  synthetic_user_interruption/  *.wav + metadata *.json
```

Metadata JSON is required for turn-taking (`turn_taking` field) and user
interruption (`interrupt` field). Place it next to the wav as `<stem>.json`, or
as `<stem>_meta.json` / `metadata/<stem>.json`.

## Install

From the `moshi/` package directory:

```bash
pip install -e '.[eval]'
```

## Generate

```bash
python -m moshi.evaluate.run_bench \
  --bench-root /path/to/fullduplex_bench \
  --results-root results/fullduplex \
  --hf-repo kyutai/moshiko-pytorch-bf16 \
  --device cuda
```

Or: `moshi-eval-bench --bench-root ...`

Optional: `--dataset-names candor_pause_handling icc_backchannel`, `--skip-existing`.

## Score

Word timestamps use NVIDIA Parakeet (GPU). User Interruption also needs an
OpenAI-compatible LLM:

```bash
export LLM_BASE_URL=https://api.openai.com/v1   # or OpenRouter / local vLLM
export LLM_API_KEY=...
export LLM_MODEL_NAME=gpt-4-turbo               # optional

python -m moshi.evaluate.evaluate \
  --results-root results/fullduplex
```

Or: `moshi-eval --results-root results/fullduplex`

Per-sample scores: `{stem}.score.json`. Per-folder summary: `score_summary.json`
with mean `TO`, `latency`, `js_divergence`, `freq`, `score` as applicable.

If `LLM_API_KEY` / `LLM_BASE_URL` are missing, User Interruption is skipped; the
other four categories still run.

## Notes

- Take-over (`TO`) follows FullDuplexBench: speech span ≥ 1s **or** ≥ 3 words.
- Backchannel compares Silero-VAD backchannel timing to the ICC ground-truth
  histogram via Jensen–Shannon divergence.
- This package does **not** include QA / RAG / Behavior judges.
