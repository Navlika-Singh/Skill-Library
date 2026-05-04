# ManaMind Skill Library

A lightweight "skill retrieval" mechanism for a progression bot for QA testing of games.

## Setup

Create the conda environment:

```bash
conda env create -f env.yaml
conda activate manamind
```

(Optional) Configure GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

## Running

Run the provided script:

```bash
bash run.sh
```

Or run manually:

```bash
python main.py \
  --json_path assets/skills.json \
  --db_path assets/skills.db \
  --top_categories 2 \
  --top_skills 3 \
  --task "The agent has low health and needs to buy a health potion"
```

## Example Tasks

- The agent has low health and needs to buy a health potion  
- The agent is fighting the final boss and has very low health  
- The agent needs to buy something but has no money  

## Example Outputs

Logs for the example runs are available in the `results/` directory:

```
results/
├── example_1.txt
├── example_2.txt
├── example_3.txt
```

Each file contains the full output of a corresponding run from `run.sh`, including retrieved skills, scores, and LLM-selected skill.

## Arguments

- `--task`: Task description  
- `--json_path`: Path to skills JSON file  
- `--db_path`: Path to SQLite database  
- `--top_categories`: Number of categories to consider  
- `--top_skills`: Number of skills to retrieve  

## Project Structure

```text
Skill-Library/
└── ManaMind/
    ├── assets/
    ├── module/
    ├── buy_item.json
    ├── main.py
    ├── requirements.txt
    └── run.sh
```