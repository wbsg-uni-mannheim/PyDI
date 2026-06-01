# Upstream Reference

Intended upstream repository:
- https://github.com/megagonlabs/ditto

This workspace environment cannot access GitHub directly (DNS/network restricted), so this directory currently contains a modernized in-repo Ditto-style training stack rather than a direct git mirror.

To vendor upstream Ditto later on a networked machine:

```bash
git clone https://github.com/megagonlabs/ditto.git /tmp/ditto_upstream
```

Then copy/adapt into `third_party/ditto_modern/` while preserving this repo's training/data CLI interfaces in `scripts/ditto/`.
