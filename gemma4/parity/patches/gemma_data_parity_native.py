"""Native-template variant of the parity data processor: render the prompt with the tokenizer's
own gemma-4 chat template (apply_chat_template -> the canonical <|turn> markers), instead of the
manual gemma-3-style <start_of_turn> markers. Used to probe whether gemma4-e2b-raw is the right
instruct model that simply needs the native markers (gemma4-e2b-raw scored ~0-2% under the
gemma-3-style render). Matches the GPU contract: system + user, apply_chat_template once,
add_generation_prompt=True. Gold/#### extraction reused from gemma_data_parity.
"""

import json
import os
import sys

# MaxText's load_custom_callable execs this file without /workload/patches on sys.path, so the
# sibling import below fails unless we add this file's dir first (same fix as the reward adapter).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import gemma_data_parity as _base  # noqa: E402 (sibling module mounted in the same dir)

_PRINTED = [False]


def process_data(dataset_name, model_tokenizer, template_config, tmvp_config, x):
    system_text, question, gold = _base._extract(x)
    try:
        prompt = model_tokenizer.apply_chat_template(
            [{"role": "system", "content": system_text},
             {"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        # Some gemma chat templates reject a system role; merge it into the user turn.
        prompt = model_tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{system_text}\n\n{question}"}],
            tokenize=False, add_generation_prompt=True)
    # tunix's tokenize() adds BOS; strip a template-prepended BOS to avoid a double BOS.
    for bos in ("<bos>", "<|bos|>", "<|begin_of_text|>"):
        if prompt.startswith(bos):
            prompt = prompt[len(bos):]
            break
    answer_field = json.dumps([gold])
    if not _PRINTED[0]:
        print("PARITY-PROMPT %r" % prompt[:320], flush=True)
        print("PARITY-GOLD   %r" % answer_field, flush=True)
        _PRINTED[0] = True
    return {"prompts": prompt, "question": question, "answer": answer_field}
