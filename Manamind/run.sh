#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

#example 1
# python main.py --json_path assets/skills.json --db_path assets/skills.db \
# --top_categories 2 --top_skills 3 --task "The agent has low health and has no health potion, so needs to buy a health potion" \

#example 2
# python main.py --json_path assets/skills.json --db_path assets/skills.db \
# --top_categories 2 --top_skills 3 --task "The agent needs to fight the final boss with the most powerful weapon possible" \

#example 3
# python main.py --json_path assets/skills.json --db_path assets/skills.db \
# --top_categories 2 --top_skills 3 --task "The agent is in middle of a fight with the final boss and the agent punches the boss, what should the agent do next?" \

#example 4
# python main.py --json_path assets/skills.json --db_path assets/skills.db \
# --top_categories 2 --top_skills 3 --task "The agent is in middle of a fight with the final boss and the agent's health is very low" \

#example 5
# python main.py --json_path assets/skills.json --db_path assets/skills.db \
# --top_categories 2 --top_skills 3 --task "The agent needs to buy something but its wallet is empty" \

#example 6
# python main.py --json_path assets/skills.json --db_path assets/skills.db \
# --top_categories 2 --top_skills 3 --task "The agent is lost and needs to find the way back" \