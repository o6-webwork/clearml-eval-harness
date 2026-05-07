# data/

This directory holds the eval JSONL source files. They are not tracked in
version control (see `.gitignore`) — sync them from the canonical host before
running evals.

## Required files

| File | Description |
|------|-------------|
| `mono_sea.jsonl` | Monolingual SEA corpus (Indonesian, Malay, Thai, Vietnamese) |
| `mono_me.jsonl` | Monolingual Middle East corpus |
| `mono_sea_id_slang.jsonl` | Indonesian slang split (generated from `mono_sea.jsonl` via `prepare_slang_split.py`) |
| `mono_sea_id_formal.jsonl` | Indonesian formal split (generated alongside `mono_sea_id_slang.jsonl`) |

## How to get them

**From a canonical host on the Tailscale mesh:**

```bash
CANONICAL_HOST=<HOST_TAILSCALE_IP> bash clearml_sync_data.sh
# or data only (skip the large HF model cache):
CANONICAL_HOST=<HOST_TAILSCALE_IP> SYNC_DATA_ONLY=1 bash clearml_sync_data.sh
```

**Generate the slang/formal splits yourself** (once `mono_sea.jsonl` is present):

```bash
python prepare_slang_split.py     # writes data/mono_sea_id_slang.jsonl and data/mono_sea_id_formal.jsonl
```

## Tracked files

`build_kamus_alay.py` and `kamus_alay.json` are checked in — they are the
slang lexicon builder and the pre-built lexicon used by `prepare_slang_split.py`
and `slang_density.py`.
