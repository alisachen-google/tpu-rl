"""GEMMA4 stop fix part 2: the continuous-decode DEVICE loop stops rows via runner.eos_token_id
(runner_utils.get_eos_token_id(model_config)) -- the tunix/vLLM host-side stop_token_ids are
irrelevant inside a CD window. tianyu g4-001d proved it: with stop ids [1,50,106] in
sampling_params only, step 0 was still 99.46% capped. Union EXTRA_STOP_TOKEN_IDS into
runner.eos_token_id (CD masking and host extract accept int-or-list). Ported from
tianyu-rl-g4-003. Prints RUNNER-EOS PATCH: APPLIED|FAILED and RUNNER-EOS ids=[...] at init."""
import sys
P = "/usr/local/lib/python3.12/site-packages/tpu_inference/runner/tpu_runner.py"
s = open(P).read()
if "_RUNNER_EOS" in s:
    print("RUNNER-EOS PATCH: already applied", flush=True); sys.exit(0)
a = "        self.eos_token_id = runner_utils.get_eos_token_id(self.model_config)\n"
if s.count(a) != 1:
    print("RUNNER-EOS PATCH: FAILED anchor %d" % s.count(a), flush=True); sys.exit(0)
b = ("        self.eos_token_id = runner_utils.get_eos_token_id(self.model_config)\n"
     "        # _RUNNER_EOS: union env EXTRA_STOP_TOKEN_IDS (chat-turn enders, e.g. gemma 106/50) into the CD stop set\n"
     "        import os as _re_os, json as _re_json\n"
     "        import numpy as _re_np\n"
     "        try:\n"
     "            _re_extra = [int(x) for x in _re_json.loads(_re_os.environ.get('EXTRA_STOP_TOKEN_IDS', '[]'))]\n"
     "        except Exception:\n"
     "            _re_extra = []\n"
     "        if _re_extra:\n"
     "            _re_base = self.eos_token_id\n"
     "            if _re_base is None:\n"
     "                _re_ids = _re_extra\n"
     "            elif isinstance(_re_base, (int, _re_np.integer)):\n"
     "                _re_ids = [int(_re_base)] + _re_extra\n"
     "            else:\n"
     "                _re_ids = [int(x) for x in _re_np.atleast_1d(_re_base).tolist()] + _re_extra\n"
     "            self.eos_token_id = tuple(sorted(set(_re_ids)))\n"
     "            print('RUNNER-EOS ids=%s' % (self.eos_token_id,), flush=True)\n")
s = s.replace(a, b)
compile(s, P, "exec"); open(P, "w").write(s); print("RUNNER-EOS PATCH: APPLIED", flush=True)
