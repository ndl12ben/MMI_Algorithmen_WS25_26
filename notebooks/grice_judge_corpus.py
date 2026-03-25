#!/usr/bin/env python3
"""
Grice-based coherence scoring over a MultiWOZ-style dialogue corpus.

This script is a notebook-to-script conversion intended for server runs.
It:
- loads a JSON dialogue corpus (e.g., MultiWOZ data.json-like structure)
- iterates over ALL dialogs (or a specified subset)
- scores EACH turn using an Ollama-served LLM
- writes results as JSONL (one row per information unit)
- writes parse errors as JSONL rows with an _error field

Usage examples:
  python grice_judge_corpus.py \
    --data-path ../data/raw/data.json \
    --out-jsonl outputs/grice_full.jsonl \
    --host http://intern.schlaubox.de:11434 \
    --model llama3:70b

  # Evaluate only selected dialog ids (one per line)
  python grice_eval_corpus.py \
    --data-path ../data/raw/data.json \
    --out-jsonl outputs/grice_subset.jsonl \
    --dialog-ids-file common_dialog_ids.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from ollama import Client
except ImportError as e:
    raise SystemExit(
        "Missing dependency: 'ollama'. Install with: pip install ollama\n"
        f"Import error: {e}"
    )


# -----------------------
# Prompts
# -----------------------
SYS_PROMPT = r"""
You are a dialogue COHERENCE judge for task-oriented dialogues.

Your task is to evaluate discourse coherence, not task success.
The dialogues are human-human, scripted interactions created for a dataset.

Judge utterances using a task-specific interpretation of Grice's Cooperative Principle:

- Quality: The utterance does not contradict prior dialogue and handles information consistently.
- Quantity: The utterance provides an appropriate amount of information for the interaction.
- Relation: The utterance is relevant to the preceding dialogue context.
- Manner: The utterance is clear, concise, and unambiguous as a conversational move.

Do NOT evaluate real-world correctness.
Do NOT assume access to external knowledge.
All utterances are to be treated as text within a dialogue, not as factual claims.
"""
GRICE_ROLE_AWARE_PROMPT = Template(r"""
Task:
Score the NEXT turn only.
Judge whether the next turn shows cooperative, role-appropriate behaviour
in a task-oriented dialogue between a CUSTOMER and an EXPERT.

Definition:
Cooperative behaviour concerns how clearly and appropriately a speaker
signals relevance, intent, and alignment with the ongoing interaction.
This judgment is grounded in Grice's Cooperative Principle,
interpreted at the discourse level (not task success).

Role expectations:

CUSTOMER (cooperative participation):
- Expresses an interpretable goal or response (Quality)
- Provides relevant information when appropriate (Relation)
- Is consistent or clearly signals corrections (Quantity)
- May be vague unless vagueness blocks progress (Manner)

EXPERT (assistance behaviour; coherence-scoped):
- Handles information consistently without contradiction (Quality)
- Responds to CUSTOMER intent at the discourse level (Relation)
- Provides as much information as needed for the interaction (Quantity)
- Is clear enough to function as a conversational move (Manner)

Scoring:

Neutral behaviour:
Turns that are interactionally appropriate but task-inert
(e.g., greetings, acknowledgments, closings).
Neutral turns are NOT failures unless they obstruct or mislead.

Labels:
Very poor | Poor | Weak | Neutral | Good | Very good | Excellent

Notes:
- For CUSTOMER turns, scores reflect cooperative participation, not assistance quality.
- Do not over-reward politeness or verbosity.
- Reserve “Excellent” for contextually outstanding behaviour.

Instructions:
- Score ONLY the next turn.
- If the turn contains multiple information units, split them.
- Split only at sentence boundaries or clearly separate requests/claims.
- Reduce each unit to its task-relevant core.
- Copy the exact PHRASE from the next turn.
- "related_text": MUST copy the most relevant earlier phrase. Use "" only if the next turn is completely unrelated to the dialogue so far.
- Prefer the most recent relevant one.
- Keep reasoning to MAXIMUM 5 words.

Return ONLY a JSON array. No surrounding text.

Each array item must be:
{
  "text": "<phrase from next turn>",
  "related_text": "<phrase from earlier, related unit or ''>",
  "reasoning": "<≤5 words>",
  "score": "<one label>"
}

Dialogue so far:
$dialogue

Next turn:
$last_utterance
""".strip())


# -----------------------
# Helpers
# -----------------------
def safe_json_loads(s: str) -> Tuple[bool, Any]:
    """Parse model output; return (ok, parsed_or_error_str).

    :param s: Raw model output string.
    :return: (ok, parsed) if ok else (False, error_message).
    """
    try:
        parsed = json.loads(s)
        if not isinstance(parsed, list):
            return False, "Model output is not a JSON array (expected list)."
        return True, parsed
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def turn_speaker(turn_index: int) -> str:
    """Infer speaker for MultiWOZ-style alternating logs (user first, then system).

    :param turn_index: Index in dialog['log'].
    :return: 'customer' for even indices else 'expert'.
    """
    return "customer" if (turn_index % 2 == 0) else "expert"


def iter_dialog_ids(
    data: Dict[str, Any],
    dialog_ids_file: Optional[Path],
    limit: Optional[int],
) -> List[str]:
    """Resolve dialog ids to score.

    :param data: Corpus dict keyed by dialog_id.
    :param dialog_ids_file: Optional file with one dialog_id per line.
    :param limit: Optional maximum number of dialogs.
    :return: List of dialog ids to process.
    """
    if dialog_ids_file:
        ids: List[str] = []
        for line in dialog_ids_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line in data:
                ids.append(line)
        if not ids:
            raise ValueError(f"No dialog ids from {dialog_ids_file} matched keys in data.")
    else:
        ids = list(data.keys())

    if limit is not None:
        ids = ids[: max(0, limit)]
    return ids


@dataclass
class RunConfig:
    data_path: Path
    out_jsonl: Path
    host: str
    model: str
    timeout: int
    limit: Optional[int]
    dialog_ids_file: Optional[Path]
    flush_every: int
    sleep_s: float


# -----------------------
# Core scoring
# -----------------------
def score_corpus(cfg: RunConfig) -> None:
    """Run Grice scoring over a full corpus and write JSONL.

    :param cfg: Run configuration.
    :return: None
    """
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"Missing: {cfg.data_path.resolve()}")

    cfg.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with cfg.data_path.open(encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    dialog_ids = iter_dialog_ids(data, cfg.dialog_ids_file, cfg.limit)

    print("DATA_PATH:", cfg.data_path.resolve())
    print("OUT_JSONL:", cfg.out_jsonl.resolve())
    print("Dialogs in file:", len(data))
    print("Dialogs to process:", len(dialog_ids))
    print("Ollama host:", cfg.host)
    print("Model:", cfg.model)
    sys.stdout.flush()

    client = Client(host=cfg.host, timeout=cfg.timeout)

    # Small warm-up call to fail fast if host/model is wrong.
    try:
        _ = client.generate(model=cfg.model, prompt="ping", options={"num_predict": 1})
    except Exception as e:
        raise RuntimeError(f"Ollama warm-up failed for host={cfg.host} model={cfg.model}: {e}")

    total_units = 0
    total_errors = 0
    start_t = time.time()

    with cfg.out_jsonl.open("w", encoding="utf-8") as f_jsonl:
        for d_i, dialog_id in enumerate(dialog_ids, start=1):
            dialog = data[dialog_id]
            log = dialog.get("log", [])

            dialogue_so_far = ""
            for t_i, turn in enumerate(log):
                text = (turn.get("text") or "").strip()
                speaker = turn_speaker(t_i)
                last_utterance = f"{speaker}: {text}\n"

                messages = [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": GRICE_ROLE_AWARE_PROMPT.substitute(
                        dialogue=dialogue_so_far,
                        last_utterance=last_utterance,
                    )},
                ]

                try:
                    resp = client.chat(model=cfg.model, messages=messages)
                    llm_output = resp["message"]["content"]
                except Exception as e:
                    total_errors += 1
                    err_record = {
                        "_dialog_id": dialog_id,
                        "_turn_index": t_i,
                        "_speaker": speaker,
                        "_source_text": text,
                        "_error": f"CHAT_CALL_FAILED: {type(e).__name__}: {e}",
                    }
                    f_jsonl.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                    dialogue_so_far += last_utterance
                    continue

                ok, parsed_or_err = safe_json_loads(llm_output)

                if not ok:
                    total_errors += 1
                    err_record = {
                        "_dialog_id": dialog_id,
                        "_turn_index": t_i,
                        "_speaker": speaker,
                        "_source_text": text,
                        "_error": parsed_or_err,
                        "_raw_output_first400": llm_output[:400],
                    }
                    f_jsonl.write(json.dumps(err_record, ensure_ascii=False) + "\n")
                else:
                    parsed = parsed_or_err
                    for unit_i, unit in enumerate(parsed):
                        if not isinstance(unit, dict):
                            continue
                        unit_record = {
                            **unit,
                            "_dialog_id": dialog_id,
                            "_turn_index": t_i,
                            "_unit_index": unit_i,
                            "_speaker": speaker,
                            "_source_text": text,
                        }
                        f_jsonl.write(json.dumps(unit_record, ensure_ascii=False) + "\n")
                        total_units += 1

                dialogue_so_far += last_utterance

                if cfg.sleep_s > 0:
                    time.sleep(cfg.sleep_s)

            if (d_i % cfg.flush_every) == 0:
                f_jsonl.flush()
                elapsed = time.time() - start_t
                rate = d_i / elapsed if elapsed > 0 else 0.0
                print(f"[{d_i}/{len(dialog_ids)}] dialogs done | units={total_units} errors={total_errors} | {rate:.2f} dlg/s")
                sys.stdout.flush()

    elapsed = time.time() - start_t
    print("\nDone.")
    print("Dialogs processed:", len(dialog_ids))
    print("Units written:", total_units)
    print("Error rows:", total_errors)
    print(f"Elapsed: {elapsed:.1f}s")
    print("Wrote:", cfg.out_jsonl.resolve())


def build_argparser() -> argparse.ArgumentParser:
    """Create CLI parser.

    :return: Configured ArgumentParser.
    """
    p = argparse.ArgumentParser(description="Run Grice scoring over a dialogue corpus and write JSONL.")
    p.add_argument("--data-path", type=Path, required=True, help="Path to corpus JSON (dict[dialog_id] -> dialog).")
    p.add_argument("--out-jsonl", type=Path, required=True, help="Output JSONL path.")
    p.add_argument("--host", type=str, default="http://intern.schlaubox.de:11434", help="Ollama host URL.")
    p.add_argument("--model", type=str, default="llama3:70b", help="Ollama model name.")
    p.add_argument("--timeout", type=int, default=500, help="Ollama client timeout in seconds.")
    p.add_argument("--dialog-ids-file", type=Path, default=None, help="Optional file with dialog IDs to process.")
    p.add_argument("--limit", type=int, default=None, help="Optional max number of dialogs (for testing).")
    p.add_argument("--flush-every", type=int, default=25, help="Flush + progress print every N dialogs.")
    p.add_argument("--sleep-s", type=float, default=0.0, help="Optional sleep between turns (throttling).")
    return p


def main() -> None:
    """CLI entry point."""
    args = build_argparser().parse_args()

    cfg = RunConfig(
        data_path=args.data_path,
        out_jsonl=args.out_jsonl,
        host=args.host,
        model=args.model,
        timeout=args.timeout,
        limit=args.limit,
        dialog_ids_file=args.dialog_ids_file,
        flush_every=max(1, args.flush_every),
        sleep_s=max(0.0, args.sleep_s),
    )
    score_corpus(cfg)


if __name__ == "__main__":
    main()
