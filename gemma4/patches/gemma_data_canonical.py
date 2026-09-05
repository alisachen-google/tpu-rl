"""gemma4 GPU-parity prompt builder (dataset_processor_path hook) — g4r41 probe.

WHY. The GPU verl run (rl05_parity_gemma4, 2026-09-02) scores ~55% correct where our TPU
runs score 2-3%, with reward fn / sampling / hyperparams / checkpoint source all verified
identical. The one divergent pipeline step: the FINAL PROMPT RENDERING. verl applies the
model's CANONICAL gemma-4 chat template at dataset load (run_gemma4_e2b_rl05_parity.sh:
"verl applies the model's chat template at load time"), so the GPU model sees its trained
turn frame <|turn>(id 105) / <turn|>(id 106) wrapped AROUND the parquet's textual
gemma-3-style content. Our TPU prompt has only the inner textual string — no trained frame.
Their exact template rendering (chat_template.jinja lines 188/234/372/382):
    <bos><|turn>user\n{content}<turn|>\n<|turn>model\n
with content = "<start_of_turn>user\n{SYS}\n\n{Q}<end_of_turn>\n<start_of_turn>model".

WHAT. Reproduce the GPU prompt byte-exactly (bos still added by tunix tokenize()).
Model should then emit <turn|>=106 to close its turn -> natural stop via
EXTRA_STOP_TOKEN_IDS [106,50]; quality should jump toward GPU's ~55% if the prompt
contract is the root cause (vs a checkpoint-conversion defect, which this probe would
show as still-broken output under the canonical frame).

SELF-CHECK. On first call, prints the token ids of "<|turn>" and "<turn|>" through the
run's tokenizer: single ids [105]/[106] confirm the trained-frame parity; a multi-piece
split means the tokenizer is not parsing added tokens and the probe is invalid."""
import json

SYSTEM_PROMPT = ("You are given a problem. Think about the problem and provide your reasoning. Place it between "
                 "<reasoning> and </reasoning>. Then, provide the final answer (i.e., just one numerical value) "
                 "between <answer> and </answer>.")

_PRINTED = [False]


def _s(v):
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def process_data(dataset_name, model_tokenizer, template_config, tmvp_config, x):
    from maxtext.trainers.post_train.rl import utils_rl
    for key in ("problem", "prompt", "question"):
        if key in x:
            question = _s(x[key]); break
    for key in ("answer", "solution", "expected_answer"):
        if key in x:
            answer = _s(x[key]); break
    qt = _s(x["question_type"]) if "question_type" in x else "default"
    processed_answer = utils_rl.process_answer(question, answer, qt)
    inner = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{question}<end_of_turn>\n<start_of_turn>model"
    prompt = f"<|turn>user\n{inner}<turn|>\n<|turn>model\n"
    if not _PRINTED[0]:
        print("GEMMA-CANON-PROMPT %r" % prompt[:300], flush=True)
        try:
            for probe in ("<|turn>", "<turn|>"):
                ids = None
                for enc in ("encode", "tokenize", "__call__"):
                    fn = getattr(model_tokenizer, enc, None)
                    if fn is None:
                        continue
                    try:
                        r = fn(probe)
                        ids = r["input_ids"] if isinstance(r, dict) else r
                        break
                    except Exception:
                        continue
                print("GEMMA-CANON-IDS %r -> %s (single trained id expected: 105/106)" % (probe, ids), flush=True)
        except Exception as e:
            print("GEMMA-CANON-IDS check failed: %r" % (e,), flush=True)
        _PRINTED[0] = True
    return {"prompts": prompt, "question": question, "answer": json.dumps(processed_answer)}
