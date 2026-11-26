# Results

Every run and every score behind this tutorial. Each file is one day, saved as JSON, and each folder has a
`summary.json`.

| Folder | What it holds | Made by |
|---|---|---|
| `compare/<model>/` | the model comparison: 5 models on 38 past days, 190 runs, with each answer, its tokens, its time and its cost | `eval/compare.py --all-event-days`, then `eval/summarize.py` |
| `judge/` | Claude Opus as the AI judge, scoring the 5 answers for each of the 38 days without seeing model names | `eval/judge.py` |
| `judge-kimi-k2.6/` | Kimi K2.6 on CoreWeave as the second judge, on the 18 days where its output was valid | `eval/judge.py --judge moonshotai/Kimi-K2.6` |
| `local/qwen3.5_4b/` | the same requests sent to Qwen3.5 4B in Ollama on a laptop | `eval/local_replay.py` |
| `compare-recorded/`, `judge-recorded/`, `local-recorded/` | a smaller 3-day run of the same scripts | the same scripts |
| `compare-train/<model>/` | for the [fine-tuning tutorial](../../Fine-Tune%20an%20Open%20Model%20Tutorial/): four open models on 121 earlier market days (Jan 2 to Aug 21, 2026), 484 runs | `eval/compare.py` with `COMPARE_OUT=results/compare-train` |
| `judge-train-glm-5.3/` | GLM 5.3 on CoreWeave scoring those 484 answers; the 5 out of 5 answers became the training data | `eval/judge.py --judge zai-org/GLM-5.3 --tries 4` |
| `judge-train-kimi/`, `judge-train-minimax-m3/` | the judges we tested on 4 training days before picking GLM 5.3 (Kimi returned no scores or all 5s) | the same script |
| `compare-test/<model>/` | base Nemotron and the two fine-tuned checkpoints on the 38 test days | `eval/compare.py` with `COMPARE_OUT=results/compare-test` |
| `judge-test-opus/`, `judge-test-glm-5.3/` | both judges scoring the test runs blind: wrong calls 5 → 2 with the 3-epoch checkpoint | `eval/judge.py` |
| `archive/` | earlier versions, kept so you can see what changed (see below) | |

## The numbers

| Model | Cost per run | Opus as judge (rank) | Kimi as judge (rank) | Named every big move | Quiet on paperwork days |
|---|---|---|---|---|---|
| Claude Opus 5.5 | $0.0251 | 4.83 (1st) | 4.00 (3rd) | 13 of 13 | 9 of 9 |
| DeepSeek V4.1 Flash | $0.0013 | 4.67 (2nd) | 3.94 (4th) | 13 of 13 | 9 of 9 |
| **Nemotron 3.5 Lightning** | **$0.0009** | **4.22 (3rd)** | **4.44 (2nd)** | 13 of 13 | 9 of 9 |
| Qwen3.6 35B | $0.0011 | 4.17 (4th) | 3.89 (5th) | 11 of 13 | 9 of 9 |
| gpt-oss-120b | $0.0002 | 4.11 (5th) | 4.50 (1st) | 13 of 13 | 9 of 9 |

Judge scores run from 1 to 5 and are averages over the 18 days both judges scored. The two judges picked the same
best answer on 10 of those 18 days.

## What's in `archive/`

| Folder | Why we replaced it |
|---|---|
| `compare-first-test/` | a first test on 5 days. DeepSeek failed every call until the gateway removed a request field that CoreWeave rejects |
| `compare-v1/`, `judge-v1/` | the first full run and its scores. At that point `SOUL.md` still had an example sentence and a "fell" template that the smaller models copied |
| `judge-v2-oldprompt/` | Opus scoring the fixed runs with the first version of the judge prompt. We then made the prompt stricter, spelling out the 1 to 5 scale and the answer letters, so a second judge could follow it. Both judges were run again with the new prompt |

The fixes to the agent are in `hermes/SOUL.md`. The prompt now uses placeholders instead of example numbers, and the
sign of the number decides whether a stock "rose" or "fell". With the Opus judge, the fix raised Nemotron's overall
score from 3.00 to 3.42 and DeepSeek's from 4.26 to 4.55.
