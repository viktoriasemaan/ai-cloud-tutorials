#!/usr/bin/env python3
"""Step 5 of the fine-tuning tutorial: train a LoRA adapter on Nemotron with CoreWeave Serverless SFT.

Uses OpenPipe ART's ServerlessBackend: your machine only sends the data, CoreWeave's GPUs train the adapter, and every
checkpoint is saved as a W&B artifact and hosted for inference right away. Training is billed per GPU-hour of active
training ($2.70, list price Nov 2026).

  pip install openpipe-art
  export WANDB_API_KEY=$(cat ~/.config/coreweave/forge_key)      your CoreWeave Forge (W&B) key
  python3 scripts/train_sft.py --epochs 1                          a short first job, to check the cost
  python3 scripts/train_sft.py --epochs 3                          the full job

Prints the name to call the trained model with, and saves it to data/trained_model.json.
"""
import argparse, asyncio, json, os, pathlib, time

import art
from art.serverless.backend import ServerlessBackend
from art.utils.sft import train_sft_from_file

HERE = pathlib.Path(__file__).resolve().parents[1]
BASE = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="market-alerts-sft")
    ap.add_argument("--project", default="market-watch-agent")
    ap.add_argument("--data", default=str(HERE / "data" / "train.jsonl"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    a = ap.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        key = pathlib.Path.home() / ".config" / "coreweave" / "forge_key"
        os.environ["WANDB_API_KEY"] = key.read_text().strip()

    backend = ServerlessBackend()
    model = art.TrainableModel(name=a.name, run_name=a.name + "-run", project=a.project, base_model=BASE)
    await model.register(backend)
    n = sum(1 for _ in open(a.data, encoding="utf-8"))
    print(f"training {a.name} on {BASE}: {n} examples, {a.epochs} epoch(s), batch {a.batch_size}, lr {a.lr}", flush=True)
    t0 = time.time()
    await train_sft_from_file(model=model, file_path=a.data, epochs=a.epochs, batch_size=a.batch_size,
                              peak_lr=a.lr, schedule_type="cosine", warmup_ratio=0.1, assistant_turns="last", verbose=True)
    step = await model.get_step()
    info = {"base_model": BASE, "name": a.name, "project": a.project, "step": step,
            "inference_name": model.get_inference_name(), "epochs": a.epochs, "examples": n,
            "wall_seconds": round(time.time() - t0, 1)}
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "trained_model.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
