# Upstream Reference

Vendored from:

- Repository: https://github.com/VIDA-NYU/magneto-matcher
- Commit:     6620623265fc7feac0f053996e62b68a13a72a57 (main, fetched 2026-04-20)
- License:    Apache-2.0 (see [LICENSE](LICENSE))
- Vendored subtree: `algorithms/magneto/magneto/` → `magneto/`

## Vendored changes

Only import-path rewrites were applied — no logic changes. Upstream uses
absolute imports (`from magneto.xxx import ...`) that work when the
package is installed via `pip install -e algorithms/magneto/`. Those are
rewritten to relative form (`from .xxx import ...`) so the package
resolves as `usecases_synthetic.third_party.magneto_matcher.magneto`
without installation.

## Scope

- Core Magneto schema-matcher used by
  [usecases_synthetic/lib/magneto_sm_matcher.py](../../lib/magneto_sm_matcher.py).
- SLM retrieval (sentence-transformer) + optional LLM reranker
  (`LLMReranker` via `litellm`).

## Re-vendoring

To refresh against a newer upstream on a networked machine:

```bash
git clone https://github.com/VIDA-NYU/magneto-matcher.git /tmp/magneto_upstream
cp -R /tmp/magneto_upstream/algorithms/magneto/magneto/ \
    usecases_synthetic/third_party/magneto_matcher/
cp /tmp/magneto_upstream/LICENSE \
    usecases_synthetic/third_party/magneto_matcher/LICENSE
```

Then re-apply the absolute-to-relative import rewrite (see
[README.md](README.md) for the list of touched files).

## What is intentionally NOT vendored

- `algorithms/magneto/finetune/` — model fine-tuning code. The committee
  uses Magneto as-is; fine-tuning is out of scope.
- `algorithms/magneto/examples/` — upstream usage examples.
- `algorithms/magneto/setup.py`, `MANIFEST.in`, `requirements.txt` — we
  do not install Magneto as a package; imports are relative.
- `experiments/` — benchmark drivers against GDC / Valentine corpora.
