import argparse
import sys
from module.database import SkillDatabase
from module.model import EmbeddingModel, LLM

def main():
    parser = argparse.ArgumentParser(description="Skill retrieval for ManaMind agent")
    parser.add_argument("--task", type=str, help="Task description (e.g., 'Buy a potion')")
    parser.add_argument("--json_path", type=str, help="Path to skills JSON file")
    parser.add_argument("--db_path", type=str, help="Path to SQLite database file")
    parser.add_argument("--top_categories", type=int, default=3, help="Number of top categories to consider")
    parser.add_argument("--top_skills", type=int, default=5, help="Number of top skills to return")
    args = parser.parse_args()
    
    TOTAL_CATEGORIES = 5
    TOTAL_SKILLS = 10
    
    print(f"Loading embedding model...")
    embedding_model = EmbeddingModel()
    
    print(f"Loading LLM model...")
    llm_model = LLM()

    print(f"Initializing skill database from {args.json_path} -> {args.db_path}")
    db = SkillDatabase(args.db_path, args.json_path, embedding_model)

    selected_skill = None

    while not selected_skill:

        print(f"Retrieving skills for task: '{args.task}'")
        results = db.search(args.task, args.top_categories, args.top_skills)

        if not results:
            print("No skills found.")
            sys.exit(0)

        print(f"\nTop {len(results)} skills:")
        for rank, (skill_id, score, skill_dict) in enumerate(results, 1):
            print(f"\n{rank}. Skill ID: {skill_id}")
            print(f"\tScore: {score:.4f}")
            print(f"\tName: {skill_dict['name']}")
            print(f"\tCategory: {skill_dict['category']}")
            print(f"\tSuccess Rate: {skill_dict['success_rate']}")
            print(f"\tDescription: {skill_dict['description'][:200]}...")

        # Ask the LLM to select the best skill based on the retrieved candidates
        llm_prompt = """Given the following task and candidate skills, select the most appropriate skill to accomplish the task.
Task: {task}
Candidate Skills, IDs, and Descriptions:
{skills}
Respond with just the ID of the best skill.'

Output format:
<Reasoning>
Think step by step about which skill is best suited for the task, considering the descriptions and success rates. Explain your reasoning here.
</Reasoning>
<Selected Skill>
ID of the selected skill.
</Selected Skill>

If you don't find any suitable skill, respond with 'None' as the selected skill ID. You should only select a skill if it is highly relevant to the task.

Lets think step by step.
    """.format(
            task=args.task,
            skills="\n".join([f"- {s['name']} (ID: {sid}): {s['description']}" for sid, _, s in results])
        )      
    
        print("\nPrompting LLM for skill selection...")

        _, llm_response = llm_model.generate(llm_prompt, enable_thinking=False)
        
        reasoning = llm_response.split("</Reasoning>")[0].replace("<Reasoning>", "").strip()
        selected_skill = llm_response.split("</Selected Skill>")[0].split("<Selected Skill>")[-1].strip()
        
        if selected_skill == "None":
            print(f"\nLLM did not find a suitable skill for Top Categories {args.top_categories} and Top Skills {args.top_skills}.")
            print(f"\nIncreasing the number of top categories and skills to consider...")
            
            if args.top_categories == TOTAL_CATEGORIES and args.top_skills == TOTAL_SKILLS:
                print(f"\nAlready at maximum categories and skills. No suitable skill found for the task.")
                break
            
            args.top_categories += 2
            args.top_skills += 2
            
            args.top_categories = min(args.top_categories, TOTAL_CATEGORIES)  # Cap at total available categories
            args.top_skills = min(args.top_skills, TOTAL_SKILLS)  # Cap at a reasonable number of skills to consider
            
            selected_skill = None  # Reset selected skill to continue the loop
            continue
        
        print("\nLLM Reasoning:")
        print(reasoning)
        print("\nLLM Selected Skill:")
        print(selected_skill)

    db.close()

if __name__ == "__main__":
    main()