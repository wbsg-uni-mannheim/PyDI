# ditto_modern

This vendored module provides a modernized Ditto-style training/evaluation runtime for entity matching.

Scope:
- Keep Ditto-style pair-text formulation (`COL <attr> VAL <value>`)
- Use current Hugging Face / PyTorch APIs
- Support single-GPU and DDP (`torchrun`)
- Keep WDC json.gz schema compatibility

This is a pragmatic modernization layer intended for cluster training workflows.
