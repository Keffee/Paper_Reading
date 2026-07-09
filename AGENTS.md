# Repository Guidelines

## Project Structure & Module Organization
This repository combines paper-reading notes with experimental recommendation/modeling code. Root-level Markdown and HTML files, such as `论文笔记.md`, `会议速记.md`, and `行文需求.html`, hold writing drafts and research notes. Store supporting figures in `images/`. The runnable code lives under `code/mycode/`: entry scripts are `run.py`, `bert_tuning.py`, `sbert_tuning.py`, `textgnn.py`, and related tuning scripts; shared Python modules are in `code/mycode/module/`; experiment YAML files are in `code/mycode/configs/`; preprocessing notebooks and scripts are in `code/mycode/preprocess/`.

## Build, Test, and Development Commands
Create the Python environment from the exported Conda file:

```bash
conda env create -f code/torch.yml
conda activate torch
```

Run commands from `code/mycode/` so relative config and data paths resolve:

```bash
cd code/mycode
CUDA_VISIBLE_DEVICES=1 python run.py --config configs/run/bert_tuning.yml
CUDA_VISIBLE_DEVICES=1 python bert_tuning.py --config configs/bert_tuning.yml
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --config configs/run/bert_tuning.yml --distributed_train True
```

Override YAML values on the command line when doing quick tests, for example `--mode test --checkpoint ckpt/.../model.bin`.

## Coding Style & Naming Conventions
Use Python 3.8-compatible syntax, four-space indentation, and PEP 8-style formatting; `autopep8` is available in the environment. Prefer `snake_case` for functions, variables, config keys, and script names. Use `PascalCase` for `torch.nn.Module` classes such as `SimpleCLS` and `NodeClassify`. Add new datasets or collators through `module/data.py` registry helpers, and add new models through `module/models.py` plus the `solver.py` model mapping.

## Testing Guidelines
No automated test suite is currently defined. Before committing code changes, run a small smoke test with the relevant YAML and, when possible, a `--mode test` command against a known checkpoint. For notebooks, restart the kernel and run all changed cells before saving. Record key metrics, dataset slice, GPU count, and config path in the related note or PR description.

## Commit & Pull Request Guidelines
Existing history uses short summaries such as `例行更新`, `小改`, and `try`; prefer a more specific concise subject, for example `update bert tuning config` or `整理 OpenOneRec 实验结果`. Pull requests should describe the research/code change, list commands run, mention changed configs or data paths, and include screenshots only when images or rendered notes changed.

## Security & Configuration Tips
Do not commit private datasets, checkpoints, tokens, or machine-specific absolute paths unless they are intentionally documented examples. The root `.gitignore` ignores new files under `code/`, images, notebooks, and several notes; use `git add -f` only for intentional tracked artifacts.
