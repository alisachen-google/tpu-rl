"""gemma4: custom process_data (dataset_processor_path hook), ported from tianyu-rl-g4-003.
EMPIRICAL findings driving this design (theirs):
(1) the checkpoint outputs garbage under the canonical gemma-4 <|turn> markers (ids 105/106 are
    effectively untrained for it),
(2) it answers math correctly under gemma-3-STYLE textual markers (<start_of_turn>... -- 7 BPE
    pieces each, NOT special tokens in this tokenizer; teammate's 39/39-verified smoke used exactly
    that), so this builder renders the gemma-3-style prompt MANUALLY (single wrap, no
    apply_chat_template; no literal <bos> -- tunix's tokenize() lets the tokenizer add bos).
Consequence: there is no single-token turn terminator; sequences stop at <eos> (id 1), the textual
markers' final tokens via EXTRA_STOP_TOKEN_IDS ([106,50] union), or the cap."""
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
    prompt = f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\n{question}<end_of_turn>\n<start_of_turn>model\n"
    if not _PRINTED[0]:
        print("GEMMA-PROMPT %r" % prompt[:300], flush=True)
        _PRINTED[0] = True
    return {"prompts": prompt, "question": question, "answer": json.dumps(processed_answer)}
