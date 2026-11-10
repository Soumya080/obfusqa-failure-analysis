"""Batched, resumable generation with Hugging Face Transformers on a Colab T4.

Design rules (you should be able to explain each one):
1. Greedy decoding (do_sample=False) so runs are repeatable.
2. 4-bit NF4 quantization so an 8B model fits in a T4's 16 GB.
3. Left padding: decoder-only models continue from the right edge, so padding must go on the left
   in a batch, or the model would "continue" from pad tokens.
4. Every output is appended to a JSONL file on Drive under a key
   (item_id, condition, prompt hash). On restart, finished keys are skipped.
5. Only the newly generated tokens are decoded (not the prompt).
"""
import time

from .utils import append_jsonl, read_jsonl, stable_hash


def load_model(cfg, model_id=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_id = model_id or cfg["model_id"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4 has no bfloat16 support
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map="auto")
    model.eval()
    torch.manual_seed(cfg.get("seed", 0))
    print(f"[model] loaded {model_id} in 4-bit; GPU memory used ~{torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return model, tok


def chat_prompt(tok, user_text):
    """Wrap the instruction in the model's own chat template."""
    msgs = [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def make_job(item_id, condition, user_text, tok, extra=None):
    """One model call. The key is what makes resuming safe."""
    prompt = chat_prompt(tok, user_text)
    job = {
        "key": stable_hash(item_id, condition, prompt),
        "item_id": item_id,
        "condition": condition,
        "user_text": user_text,
        "prompt": prompt,
    }
    if extra:
        job.update(extra)
    return job


def run_jobs(jobs, model, tok, out_path, batch_size=16, max_new_tokens=32,
             do_sample=False, temperature=None, num_return_sequences=1):
    """Run jobs in batches and append results to out_path. Skips keys already on disk."""
    import torch

    done = {r["key"] for r in read_jsonl(out_path)}
    todo = [j for j in jobs if j["key"] not in done]
    print(f"[run] {len(jobs)} jobs, {len(done & {j['key'] for j in jobs})} already done, {len(todo)} to run -> {out_path}")
    t0 = time.time()
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        enc = tok([j["prompt"] for j in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                          pad_token_id=tok.pad_token_id, num_return_sequences=num_return_sequences)
        if do_sample:
            gen_kwargs["temperature"] = temperature
        else:  # silence warnings from generation configs that set sampling params
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        n_in = enc["attention_mask"].sum(dim=1).tolist()
        rows = []
        for bi, j in enumerate(batch):
            outs = texts[bi * num_return_sequences:(bi + 1) * num_return_sequences]
            rec = {k: v for k, v in j.items() if k != "prompt"}
            rec["n_input_tokens"] = int(n_in[bi])
            if num_return_sequences == 1:
                rec["raw_output"] = outs[0]
            else:
                rec["raw_outputs"] = outs
            rows.append(rec)
        append_jsonl(out_path, rows)
        done_n = start + len(batch)
        rate = done_n / max(time.time() - t0, 1e-6)
        print(f"  {done_n}/{len(todo)}  ({rate:.1f} jobs/s, ~{(len(todo) - done_n) / max(rate, 1e-6) / 60:.1f} min left)")
    return read_jsonl(out_path)


def determinism_check(jobs, model, tok, max_new_tokens=32):
    """Run the same jobs twice, one at a time, and compare outputs. Should print 'identical: N / N'."""
    import torch

    def once():
        outs = []
        for j in jobs:
            enc = tok(j["prompt"], return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=tok.pad_token_id, temperature=None, top_p=None, top_k=None)
            outs.append(tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs

    a, b = once(), once()
    same = sum(x == y for x, y in zip(a, b))
    print(f"[determinism] identical: {same} / {len(jobs)}")
    return a, b
