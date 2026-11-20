# Build an Always-On AI Agent with Hermes and CoreWeave

This tutorial builds an AI agent that runs around the clock, checks a data source every 30 minutes, and sends you a
short message when something important changes. Our example watches the stock market. The same design works for
anything you can check with a script. It's built like a production agent, with a trace of every run and an evaluation
before any model goes live.

![24/7 Market Watch Agent: how it works and where it runs](./visuals/where-it-runs.gif)

## What you'll learn

You will learn by building a working agent, one part at a time:

1. **How an always-on agent works.** You'll see why a free check runs before the model, why every model call goes
   through a gateway, and what each part costs. ([Part 1](#part-1--learn-how-an-always-on-agent-works))
2. **How to build one.** You'll connect your machine, CoreWeave and Telegram into one system, and get a real alert on
   your phone. ([Part 2](#part-2--build-the-agent))
3. **How to choose a model.** You'll compare five models on the same 38 market days and score them with two AI
   judges. ([Part 3](#part-3--choose-the-model-with-two-ai-judges))
4. **How to check that it's safe and cheap.** You'll try to trick your own agent with fake filings, and add up what
   every run costs. ([Part 4](#part-4--test-its-safety-and-cost))

Along the way, you'll get hands-on experience with CoreWeave Forge: Serverless Inference, the Playground, Agent Lens
traces and Insights, and Billing.

## What it does

Every 30 minutes, the agent looks at the 10 largest companies in the S&P 500 and compares each one with the S&P 500
itself. If a stock moved much more than the market, or a company filed an important report, the agent writes a short
explanation and sends it to you on Telegram. If nothing happened, it stays quiet and costs nothing.

Here is a real message it sent for October 29, 2026:

```text
Amazon fell 4.25% vs S&P 500 fell 0.38%
Alphabet fell 12.64% vs S&P 500 fell 0.38%
Eli Lilly rose 6.63% vs S&P 500 fell 0.38%
```

The agent only explains what happened. It never trades and never tells you to buy or sell. You decide what to do.

> Not financial advice.

## Use it to watch anything

The stock market is only the example. You can point the same agent at any source you can read with a script, such
as the price of a product, new home listings in your area, a status page, or new releases of a software project.
To change what it watches, you change four files:

| File | What you change |
|---|---|
| `scripts/precheck.py` | where the data comes from, and the rule that decides whether something happened |
| `config/watchlist.json` | the list of things to watch |
| `hermes/USER.md` | what you want to hear about, and what you don't |
| `hermes/SOUL.md` | how the agent should write the message |

The rest stays the same: the schedule, the agent, the gateway, the model on CoreWeave and the Telegram delivery.

## From pilot to production

Production AI runs as a loop of five stages: run, observe, curate, improve and evaluate. This tutorial covers four of
them:

| Stage | In this tutorial | Where |
|---|---|---|
| **1. Run** | Nemotron 3.5 Lightning on CoreWeave Serverless Inference answers only when the pre-check finds something | Parts 1 and 2 |
| **2. Observe** | Agent Lens traces every run, with its prompt, its answer, its time and its cost | Part 1 |
| **3. Curate** | not covered yet. Agent Lens tags are where it starts: you label runs as good or bad | Part 1 |
| **4. Improve** | the judges found two mistakes in our prompt. Fixing them raised Nemotron's score from 3.00 to 3.42 with the first Opus judge | Part 3 |
| **5. Evaluate** | two AI judges score five models on the same 38 days before one is chosen | Part 3 |

We started with one job, serving an open model, and added each stage when we needed it.

## Part 1 · Learn how an always-on agent works

In this part, you'll learn how the agent is put together: which parts run on your machine, which run on
CoreWeave, why we chose each one, and how the agent avoids calling the model when nothing has happened.

### What runs where, and what it costs

- **Your local machine** runs the schedule, the check, the agent and the gateway. This can be your laptop or a
  small cloud VM. These parts are open source and free, but the machine has to stay on.
- **CoreWeave** runs the AI model, and only when the agent calls it. CoreWeave also stores a trace of every run in
  Agent Lens.
- **Your phone** receives the message through a Telegram bot that answers only you.

| Part | Cost per month |
|---|---|
| Schedule, check, agent and gateway | $0, open source on your machine |
| Market data (Alpha Vantage free plan, SEC EDGAR) | $0 |
| AI model (Nemotron 3.5 Lightning on CoreWeave) | about 1 to 5 cents |
| Agent Lens traces | $0 at this volume |
| Telegram | $0 |
| An always-on machine, if you don't use your laptop | a small cloud VM costs a few dollars |

The model is called about 13 times a month, at about $0.0009 per call. The same job on Claude Opus would cost about
$0.33 a month. If the agent called the model on every 30-minute run instead of only when needed, it would cost about
$0.70 a month on Nemotron and about $27 on Opus.

### The parts, and why we chose them

![The always-on agent: the architecture](./visuals/architecture.png)

#### Hermes Agent: the agent harness

[Hermes Agent](https://github.com/NousResearch/hermes-agent) is an open-source agent harness from Nous Research.
A harness is the program around the model. It decides when to run, what goes into the prompt, which tools the model
may use, and where the answer goes. We chose Hermes for this agent for these reasons:

- **It runs on your own machine.** It is free and open source, so the always-on part costs nothing.
- **It has a scheduler with a script hook.** A Hermes cron job can run a script before every run. If the script
  prints `{"wakeAgent": false}`, Hermes skips the model for that run. This is what keeps the agent cheap.
- **It has memory.** The agent remembers what it already told you, and `USER.md` holds your preferences.
- **It works with any OpenAI-compatible provider.** We point it at our gateway, and the gateway sends the call to
  CoreWeave. Any model on CoreWeave works without changing the agent.
- **It delivers to Telegram.** The Telegram connection is built in, with an allowlist so the bot talks to one chat.
- **It has safety controls.** Hermes scans every prompt for prompt injection before it reaches a model, and each
  platform can be limited to a small set of tools. Ours can use only memory and session search.

Hermes is designed to run on a schedule and talk to you through a messaging app, which is what this agent needs.

#### Calling the model only when needed

Most 30-minute checks find nothing new. Calling a model every time would cost money for no reason. So a small Python
script, `scripts/precheck.py`, runs first. It uses no AI:

1. It gets daily closing prices from Alpha Vantage (at most 11 calls a day on the free plan).
2. It gets new filings from SEC EDGAR, and drops routine paperwork.
3. A price move counts only if the stock beat the S&P 500 by at least 2 points, and by more than twice its usual
   daily move.
4. If nothing counts, it prints `{"wakeAgent": false}` and Hermes skips the model.
5. If something counts, it prints the evidence as JSON, and Hermes adds it to the prompt.

Over 60 trading days, the model was needed on 38 of them. The script skipped 281 routine filings for free.

![Inside one run](./visuals/one-run.png)

#### The gateway: one address for every model

The gateway, `gateway/proxy.py`, is about 150 lines of Python that we wrote for this tutorial. Hermes talks to it as
if it were one OpenAI-compatible provider. For every call, the gateway does six things:

1. It sends the call to CoreWeave, or to Anthropic for Claude models.
2. It fixes small differences in requests that a provider would reject.
3. It writes the request and the answer to `data/calls.jsonl`.
4. It works out the cost at list prices and adds it to a monthly total.
5. It refuses new calls once the monthly budget is used up.
6. It sends a trace of the call to CoreWeave Agent Lens.

Because of the gateway, changing the model is a one-line change, and every call has a cost and a trace.

![The gateway: one address, every model](./visuals/gateway.png)

#### CoreWeave Serverless Inference: the model

[Serverless Inference](https://docs.coreweave.com/products/inference/serverless.md) gives you open models behind an
OpenAI-compatible API. You pay per token, and there is nothing to deploy or keep running. We used these parts of it:

- **Models catalog:** every model with its price per million tokens, and what it can do. We tested four open models
  from it.
- **Playground:** to try a model before writing any code. Nemotron 3.5 Lightning answered our test question in 3.9
  seconds, and you can read its reasoning.
- **View code quickstart:** the same call with the OpenAI client. Its base URL is where our gateway sends calls.
- **API key:** one key works for both inference and Agent Lens.

The model that runs our agent is **Nemotron 3.5 Lightning**. It has 30 billion parameters, but only 3 billion are
active for each token, which makes it fast and cheap: $0.07 per million input tokens and $0.20 per million output
tokens.

#### CoreWeave Agent Lens: seeing inside every run

[Agent Lens](https://docs.coreweave.com/products/agent-lens/get-started/opentelemetry.md) collects traces of agent
runs, sent as OpenTelemetry spans. It has built-in integrations for agent frameworks such as the Claude and OpenAI
Agents SDKs and Google ADK. Hermes is not one of them yet, so our gateway sends the spans itself. Each run becomes
one conversation, with an agent turn and the model call inside it. In Agent Lens you can see:

- **Overview:** activity, errors and cost by model.
- **Conversations:** one row per run. Each run has four tabs. Thread shows exactly what the model saw and wrote.
  Spans shows the structure of the run. Raw shows every attribute. Timeline shows how long each step took.
- **Insights:** LLM judges sort every turn into categories. For our agent, they flagged a missed requirement in 21%
  of turns and tool misuse in 5%. Each category links to the conversations behind it.
- **Tags:** you can label turns by hand, for example as a good alert or as one that should have stayed quiet. You can
  also set up an LLM judge that tags new turns, and Agent Lens scores that judge against your own labels.

This is the curate stage. Labeled turns become the dataset for fine-tuning the model later, in Forge's Post-Training
section.

## Part 2 · Build the agent

In this part, you'll build the agent one piece at a time and finish with a real alert on Telegram. Every step can
replay a real past day, so you can follow along on any day of the week. The examples use October 29, 2026.

![Six steps: one part at a time](./visuals/tutorial-steps.png)

### Prerequisites

- Linux, macOS or WSL with Python 3.10 or later (we used Ubuntu 24.04)
- A CoreWeave Forge account
- A free Alpha Vantage key ([alphavantage.co/support](https://www.alphavantage.co/support/))
- A Telegram bot token from @BotFather, and your chat ID
- SEC EDGAR needs no key, only a contact name and email
- Optional: an Anthropic API key for the Claude Opus comparison and judge, and Ollama for the laptop comparison

### Step 1 · Accounts and keys

Save each key in its own file under `~/.config`. Type it at a hidden prompt so it never appears on screen or in your
shell history, then make the files readable only by you:

```bash
git clone https://github.com/viktoriasemaan/ai-cloud-tutorials.git
cd "ai-cloud-tutorials/Market Watch Agent Tutorial"
mkdir -p ~/.config/coreweave ~/.config/alphavantage ~/.config/sec
read -rsp "CoreWeave API key: " K && printf %s "$K" > ~/.config/coreweave/forge_key && echo " saved (${#K} characters)"
read -rsp "Alpha Vantage key: " K && printf %s "$K" > ~/.config/alphavantage/api_key && echo " saved (${#K} characters)"
echo "Your Name you@example.com" > ~/.config/sec/user_agent
chmod 600 ~/.config/*/*
```

Every script reads an environment variable first (`FORGE_API_KEY`, `ALPHAVANTAGE_API_KEY`, `SEC_USER_AGENT`), then
the file. No key is written into this folder, and `.gitignore` blocks `.env` and `*.key` files.

### Step 2 · Install Hermes Agent

```bash
sudo apt-get install -y libatomic1        # Ubuntu only, if it's missing
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
hermes --version                          # we used 0.23.2
```

### Step 3 · The pre-check

```bash
python3 scripts/precheck.py --date 2026-10-29 --summary
```
```text
2026-10-29 (replay): 10 events
  AMZN   price   -4.25% vs S&P 500 -0.38%  (-3.86 pts, bar 2.85)
  GOOGL  price   -12.64% vs S&P 500 -0.38%  (-12.26 pts, bar 2.61)
  LLY    price   +6.63% vs S&P 500 -0.38%  (+7.01 pts, bar 3.89)
  AMZN   filing  8-K              quarterly or annual results (earnings)
  LLY    filing  8-K              quarterly or annual results (earnings)
  ...
  routine filings skipped: 5 {'4': 2, '424B2': 1, '144': 1, 'CERT': 1}
```

On a quiet day, nothing passes the bar and the model isn't called:

```bash
python3 scripts/precheck.py --date 2026-11-13 --summary
```
```text
2026-11-13 (replay): 0 events
  routine filings skipped: 1 {'FWP': 1}
  -> {"wakeAgent": false}: the model is not called
```

To run the same check over the last 60 trading days:

```bash
python3 eval/scan_days.py --end 2026-11-16
```
```text
2026-08-24 to 2026-11-16: 60 trading days
  days that wake the model:  38 of 60
  events: 33 price moves + 82 filings = 115
  routine filings skipped for free: 281  (4 192, 144 48, 424B5 12, FWP 9)
```

The free Alpha Vantage plan has closing prices only, so price moves are checked once a day after the market closes.
Filings are checked on every run.

### Step 4 · The gateway

```bash
pip install -r gateway/requirements.txt
AGENT_LENS_PROJECT=<your-team>/market-watch-agent python3 gateway/proxy.py &
```
```text
budget: $3.30 of $40.00 used this month (CoreWeave, list prices)
gateway on http://127.0.0.1:8787/v1 · CoreWeave Serverless Inference + Anthropic · Agent Lens project your-team/market-watch-agent
```

Send it a normal OpenAI-style request, then the same question with a different model name:

```bash
curl -s localhost:8787/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-oss-120b","messages":[{"role":"user","content":"In one sentence: why compare a stock move with the S&P 500?"}]}'
```
```text
18:15:14 openai/gpt-oss-120b      in 83  out 129   1.7s  $0.00002
18:15:31 claude-opus-5-5          in 29  out 75    1.5s  $0.00162
```

The same question cost about 80 times more on Claude Opus. To see the budget stop, start a second gateway with a
monthly limit of $1 (the default is $40):

```bash
GATEWAY_PORT=8790 GATEWAY_MONTHLY_LIMIT_USD=1 python3 gateway/proxy.py &
curl -s localhost:8790/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"ok"}]}'
```
```json
{"error": {"code": "gateway_budget_reached", "message": "Gateway budget reached: $3.30 of $1.00 this month. Raise GATEWAY_MONTHLY_LIMIT_USD to continue."}}
```

### Step 5 · The agent's profile

Create a separate Hermes profile with no extra skills, so the prompt stays small, and point it at the gateway:

```bash
hermes profile create marketwatch --no-skills --description 'Always-on market watch agent'
marketwatch config set model.provider custom
marketwatch config set model.base_url http://127.0.0.1:8787/v1
marketwatch config set model.default nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B
marketwatch config set model.api_key via-gateway        # the real key stays in the gateway
```

Then the safety settings. The scheduled job and the Telegram chat both get only two tools, and the bot answers one
chat:

```bash
marketwatch config set platform_toolsets.cron '["memory","session_search"]'
marketwatch config set platform_toolsets.telegram '["memory","session_search"]'
cp hermes/SOUL.md ~/.hermes/profiles/marketwatch/SOUL.md
cp hermes/USER.md ~/.hermes/profiles/marketwatch/memories/USER.md
echo "TELEGRAM_BOT_TOKEN=<token from @BotFather>" >> ~/.hermes/profiles/marketwatch/.env
echo "TELEGRAM_ALLOWED_USERS=<your chat id>"      >> ~/.hermes/profiles/marketwatch/.env
echo "TELEGRAM_HOME_CHANNEL=<your chat id>"       >> ~/.hermes/profiles/marketwatch/.env
```

`SOUL.md` holds the agent's rules. It treats the market report as data and never follows instructions inside it. It
writes only numbers that appear in the report, never tells anyone to buy or sell, and replies `[SILENT]` when nothing
is worth an alert. `USER.md` says what you want to hear about and what you don't.

### Step 6 · The first real run

Install the pre-check as a script hook for a replay of October 29, create a one-off job, start Hermes's gateway (the
scheduler and the Telegram connection), and run the job:

```bash
bash hermes/install_precheck.sh marketwatch 2026-10-29
marketwatch cron create 'every 1000d' 'Read the market report below and decide whether to alert. Follow your instructions.' \
  --name replay-2026-10-29 --script precheck_2026-10-29.sh --deliver telegram --no-wrap-response
marketwatch gateway start
marketwatch cron run <job-id>
```

The model gateway logged one call:

```text
18:21:54 nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B   in 5364 out 7041   48.2s  $0.00178
```

And the message in the example at the top of this page arrived on Telegram.

### Make it run all the time

```bash
bash hermes/install_precheck.sh marketwatch           # live mode: precheck.sh
marketwatch cron create 'every 30m' 'Read the market report below and decide whether to alert. Follow your instructions.' \
  --name market-watch-live --script precheck.sh --deliver telegram --no-wrap-response \
  --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B
marketwatch cron status
```

In live mode, the pre-check remembers which filings it already reported, so you hear about each one once.

## Part 3 · Choose the model with two AI judges

In this part, you'll learn how to pick a model with evidence instead of a price list. You'll run five models on the
same 38 days, score their answers with a script and two AI judges, and see where the judges disagree. This is the
evaluate stage: a model goes live only after it passes your own tests.

![Comparing models: only the model changes](./visuals/model-comparison.png)

We wanted to know which model should run the agent. Price per token doesn't tell you how good the messages will be,
so we tested. We ran four open models on CoreWeave, plus Claude Opus as a reference. Each model ran the same agent,
with the same instructions, on the same 38 past market days. The agent's memory was reset before every run, so the
model was the only thing that changed. That made 190 runs in total.

We scored the answers in three steps:

1. **Automatic checks.** A script checked each answer. Did it name every big price move? Did it stay quiet on days
   with only routine paperwork? Did it invent any numbers?
2. **An AI judge.** Claude Opus scored every answer from 1 to 5 for coverage, noise, accuracy and clarity. It saw all
   five answers for a day at once, in random order, labeled only with letters, so it didn't know which model wrote
   which answer.
3. **A second AI judge.** Claude Opus was also one of the models being tested, so it might favor its own answers. We
   added Kimi K2.6 on CoreWeave as a second judge, because it wasn't one of the models tested.

Using two judges mattered. They often disagreed. Opus gave higher scores to complete alerts, and Kimi preferred
short ones. gpt-oss came last with Opus as the judge and first with Kimi. The judges picked the same best answer on
only 10 of the 18 days they both scored. Kimi also broke the output format on 23 of 38 days on its first try, so
test your judge before you trust it.

![The scorecard: which model should run it](./visuals/scorecard.png)

| Model | Cost per run | Opus as judge (rank) | Kimi as judge (rank) | Named every big move |
|---|---|---|---|---|
| Claude Opus 5.5 | $0.0251 | 4.83 (1st) | 4.00 (3rd) | 13 of 13 |
| DeepSeek V4.1 Flash | $0.0013 | 4.67 (2nd) | 3.94 (4th) | 13 of 13 |
| **Nemotron 3.5 Lightning** | **$0.0009** | **4.22 (3rd)** | **4.44 (2nd)** | 13 of 13 |
| Qwen3.6 35B | $0.0011 | 4.17 (4th) | 3.89 (5th) | 11 of 13 |
| gpt-oss-120b | $0.0002 | 4.11 (5th) | 4.50 (1st) | 13 of 13 |

Judge scores are averages over the 18 days both judges scored. Every model stayed quiet on all 9 paperwork days.

We chose **Nemotron 3.5 Lightning**. It was the only open model in the top 3 for both judges, and a run costs less
than a tenth of a cent. Claude Opus costs more than 25 times as much per run.

The automatic checks also found two mistakes in our own instructions. A model copied an example sentence from the
prompt as if it were real data. Another copied the word "fell" on a day the stock went up. The larger models ignored
these mistakes, and the smaller ones copied them. The fix was to use placeholders instead of example numbers, and to
let the sign of the number decide between "rose" and "fell".

To repeat the comparison:

```bash
python3 eval/compare.py --all-event-days --models openai/gpt-oss-120b nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B \
  Qwen/Qwen3.6-35B-A3B deepseek-ai/DeepSeek-V4.1-Flash claude-opus-5-5
python3 eval/summarize.py --compact                   # automatic checks
pip install anthropic pydantic
python3 eval/judge.py                                 # Claude Opus as the judge
python3 eval/judge.py --judge moonshotai/Kimi-K2.6    # Kimi K2.6 on CoreWeave as the second judge
python3 eval/judges_table.py                          # where the two judges agree
```

## Part 4 · Test its safety and cost

In this part, you'll try to trick your own agent with fake filings, see which protection stops each attack, and
add up what the whole project cost.

### Safety

![Safety comes in layers](./visuals/safety-layers.png)

An agent that runs all the time reads text it didn't write, and that text can contain instructions. To test this, we
planted two fake filings on a real day:

```bash
bash eval/safety/run_safety.sh marketwatch
```
```text
=== plain attack
Scanner result: Blocked: prompt matches threat pattern 'prompt_injection'.
memory: nothing written
=== subtle attack
agent's answer: three lines on Amazon, Alphabet and Eli Lilly, with no trade call
memory: nothing written
preferences (USER.md): unchanged
```

The obvious attack never reached a model, because the Hermes scanner blocked it. The subtle one got past the
scanner, and the model ignored it because of the first rule in `SOUL.md`. This was one run on one model, so it shows
how the protections behave. It doesn't prove they always work, which is why the agent has several layers.

### What it cost

![What it cost to test and run](./visuals/what-it-cost.png)

```bash
python3 eval/costs.py
```

All the open-model calls together came to about 33 cents. That includes the 190 comparison runs, the tests and the
live runs. The second judge, Kimi K2.6, cost $1.26 on its own because it writes long reasoning. Running live on
Nemotron costs about a cent a month.

We also tried running the model on a laptop. `eval/local_replay.py` sends the same requests to a small local model in
Ollama. On a laptop without an NVIDIA GPU, Qwen3.5 4B took 39 seconds per decision and returned 9 empty answers out
of 38. Nemotron on CoreWeave took 9 seconds.

## What you used in CoreWeave Forge

![CoreWeave Forge: the parts we used](./visuals/forge-map.png)

By the end of this tutorial, you'll have worked with each of these parts of
[CoreWeave Forge](https://www.coreweave.com/products/coreweave-forge):

| Forge area | Stage | What you use it for |
|---|---|---|
| Inference, Playground | Run | trying Nemotron 3.5 Lightning before writing code |
| Inference, Models | Run | comparing prices and picking four open models to test |
| Inference, model page | Run | the exact model ID, and the model's size |
| Agent Lens, Overview and Conversations | Observe | seeing every run, its prompt, its answer, its tokens and its cost |
| Agent Lens, Tags | Curate | labeling good and bad alerts, by hand or with an LLM judge |
| Post-Training | Improve | the next step: fine-tuning the model on labeled runs |
| Agent Lens, Insights | Evaluate | finding patterns in failures without reading every run |
| User settings, API keys | | one key for inference and Agent Lens |
| Billing, Usage | | checking spend. Forge showed $3.30 of inference, the same as our gateway's total |

## Recommendations

1. Put a free check in front of the model, so the model runs only when something changed.
2. Send every model call through a gateway. You get one place for cost, traces and the budget, and you can change
   models without changing the agent.
3. Test models on your own past data. Write the scoring rules before you look at the results, and use more than one
   AI judge.
4. Run the cheapest model that passes your tests, and use a frontier model as the judge.

## Limits

We used one watchlist, 38 days, end-of-day prices and list prices, and our two judges didn't always agree. Use these
results as a starting point for testing your own setup. Commands and model IDs can change.

## What's in this folder

| Path | What it is |
|---|---|
| `config/watchlist.json` | the 10 companies, the S&P 500 baseline (SPY) and the alert rules |
| `scripts/precheck.py` | the check that decides whether to call the model. `--date` replays a past day |
| `gateway/proxy.py` | the gateway. `gateway/requirements.txt` adds the Agent Lens exporter |
| `hermes/SOUL.md` | the agent's rules |
| `hermes/USER.md` | your preferences |
| `hermes/precheck.sh`, `hermes/install_precheck.sh` | the script hook Hermes runs before every scheduled run |
| `eval/scan_days.py` | runs the check over the last 60 trading days |
| `eval/compare.py`, `eval/summarize.py` | runs every model on the same days, and the automatic checks |
| `eval/judge.py`, `eval/judges_table.py` | the AI judge, and the comparison of the two judges |
| `eval/safety/` | the prompt-injection test |
| `eval/local_replay.py` | the same requests sent to a local model in Ollama |
| `eval/costs.py` | what everything cost, from the gateway's log |
| `results/` | every run and every score from our tests, see [`results/README.md`](./results/README.md) |
| `visuals/` | the diagrams on this page |

## Clean up

```bash
marketwatch cron list                       # then: marketwatch cron remove <job-id>
marketwatch gateway stop
kill %1                                     # the model gateway, if you started it with &
hermes profile delete marketwatch
```

In Forge, delete the API key under User settings, and the Agent Lens project if you no longer need it.

## Learn more

- [CoreWeave Forge](https://www.coreweave.com/products/coreweave-forge) and [Forge pricing](https://coreweave.com/forge-pricing)
- [CoreWeave Serverless Inference](https://docs.coreweave.com/products/inference/serverless.md) and [its models](https://docs.coreweave.com/products/inference/serverless/models.md)
- [Agent Lens with OpenTelemetry](https://docs.coreweave.com/products/agent-lens/get-started/opentelemetry.md)
- [Hermes Agent](https://github.com/NousResearch/hermes-agent): [cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron), [memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory), [security](https://hermes-agent.nousresearch.com/docs/user-guide/security), [Telegram](https://hermes-agent.nousresearch.com/docs/user-guide/messaging/telegram)
- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
