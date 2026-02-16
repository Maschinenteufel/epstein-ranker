# Data directory

Place `EPS_FILES_20K_NOV2026.csv` (downloaded from https://huggingface.co/datasets/tensonaut/EPSTEIN_FILES_20K) in this folder.
The file is ~100 MB and is not tracked in git.

You can also point `gpt_ranker.py --input` at a directory tree of `.txt` files (for example, `data/new_data`).

For independent corpora, prefer workspace mode:
`--dataset-workspace-root data/workspaces --dataset-tag <name>`
