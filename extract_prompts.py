import json
import os

transcript_path = "/Users/deepeshkumarjha/.gemini/antigravity-ide/brain/71ec6ba4-5913-4310-8a79-1e3f86d3b6f2/.system_generated/logs/transcript.jsonl"
out_path = "benchmark_results/prompts/all_prompts.md"

with open(out_path, "w") as f:
    f.write("# Conversation Prompts\n\n")
    
    if os.path.exists(transcript_path):
        with open(transcript_path, "r") as tf:
            for line in tf:
                try:
                    data = json.loads(line)
                    if data.get("type") == "USER_INPUT":
                        content = data.get("content", "").strip()
                        if content and "CHECKPOINT" not in content and "<EPHEMERAL_MESSAGE>" not in content:
                            f.write(f"## Prompt (Step {data.get('step_index', '?')})\n")
                            f.write(f"{content}\n\n---\n\n")
                except:
                    pass
print(f"Extracted prompts to {out_path}")
