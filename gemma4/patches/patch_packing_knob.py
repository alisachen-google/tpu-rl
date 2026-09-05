"""_PACKING_KNOB v2: feed max_seq_token_per_tpu to tunix from env MAX_SEQ_TOKEN_PER_TPU.
The image's train_rl.py plumbs `max_seq_token_per_tpu=(getattr(trainer_config, "max_seq_token_per_tpu", 0) or None)`
but the key is not in MaxText's pydantic RLConfig schema, so the CLI arg is rejected. Read env instead
(0/unset = packing off). v1 bug: a '#' comment inside the replacement expression commented out
' or None),' -> unclosed paren (compile-check caught it, file untouched). v2: no inline comment;
idempotency marker appended at EOF. Fail-open."""
import glob
def _patch():
    try:
        cands = glob.glob("/deps/src/maxtext/trainers/post_train/rl/train_rl.py") + glob.glob("/app/src/maxtext/trainers/post_train/rl/train_rl.py")
        if not cands:
            print("PACKING-KNOB: train_rl.py not found (skipped)", flush=True); return
        for fp in cands:
            src = open(fp).read()
            if "_PACKING_KNOB" in src:
                print("PACKING-KNOB: already applied", fp, flush=True); continue
            anchor = 'getattr(trainer_config, "max_seq_token_per_tpu", 0)'
            n = src.count(anchor)
            if n == 0:
                print("PACKING-KNOB: anchor not found in", fp, "(skipped)", flush=True); continue
            src = src.replace(anchor, 'int(__import__("os").environ.get("MAX_SEQ_TOKEN_PER_TPU", "0"))')
            src += "\n# _PACKING_KNOB applied: max_seq_token_per_tpu read from env MAX_SEQ_TOKEN_PER_TPU\n"
            compile(src, fp, "exec"); open(fp, "w").write(src)
            import os
            print("PACKING-KNOB: train_rl reads MAX_SEQ_TOKEN_PER_TPU=%s (%d edit) in %s" % (os.environ.get("MAX_SEQ_TOKEN_PER_TPU"), n, fp), flush=True)
    except Exception as e:
        print("PACKING-KNOB: failed %r" % (e,), flush=True)
_patch()
