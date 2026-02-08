import os
import json
import requests
import time
import logging
from typing import List, Dict
from huggingface_hub import InferenceClient
from datetime import datetime

# --- Configuration ---
CHATBOT_URL = "http://localhost:5005/chat"
DATASET_PATH = "eval_dataset.json"
DELAY_BETWEEN_QUESTIONS = 5  # Seconds to wait between each evaluation step

OUTPUT_PATH = "evaluation_report2.json"
MARKDOWN_REPORT_PATH = "evaluation_report2.md"

# LLM Judge Configuration
JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
hf_token = os.getenv("HF_TOKEN") 

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not hf_token:
    logger.error("❌ HF_TOKEN not found! Inference will fail.")
    exit(1)

client = InferenceClient(api_key=hf_token)

# --- Prompts ---
JUDGE_SYSTEM_PROMPT = """You are an impartial judge evaluating a university admission chatbot's response.
Your goal is to provide three scores (0, 1, or 2) for the following criteria:

1. **Completeness**: Does the answer address everything in the question?
2. **Grounding**: Is the answer derived ONLY from the provided context (Ground Truth)?
3. **Correctness**: Is the answer factually consistent with the Ground Truth?

Score Scale:
- 0: Poor / Completely Incorrect / Hallucinated
- 1: Partially Correct / Acceptable but incomplete
- 2: Perfect / Fully correct and grounded

Your output MUST be a valid JSON object only, in this format:
{
  "completeness": score,
  "grounding": score,
  "correctness": score,
  "reasoning": "Brief explanation for the scores"
}"""

# --- Functions ---

def query_chatbot(question: str) -> Dict:
    """Send a question to the live chatbot service."""
    try:
        payload = {"query": question}
        response = requests.post(CHATBOT_URL, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Chatbot returned error {response.status_code}: {response.text}")
            return {"answer": "Error calling chatbot", "sources": []}
    except Exception as e:
        logger.error(f"Failed to connect to chatbot: {e}")
        return {"answer": "Connection failed", "sources": []}

def judge_response(item: Dict, bot_response: str) -> Dict:
    """Use LLM as a Judge to score the interaction."""
    user_prompt = f"""
QUESTION: {item['question']}
GROUND TRUTH (Reference): {item['ground_truth_answer']}
CHATBOT ANSWER: {bot_response}

Please judge the Chatbot Answer based on the Ground Truth provided.
"""
    try:
        completion = client.chat_completion(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        judge_content = completion.choices[0].message.content.strip()
        # Extract JSON if the model returns filler text
        if "{" in judge_content:
            judge_content = judge_content[judge_content.find("{"):judge_content.rfind("}")+1]
        
        return json.loads(judge_content)
    except Exception as e:
        logger.error(f"Judging failed for ID {item['id']}: {e}")
        return {"completeness": 0, "grounding": 0, "correctness": 0, "reasoning": "Judging Error"}

def run_evaluation():
    logger.info("🚀 Starting Evaluation Pipeline...")
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    results = []
    category_metrics = {}

    for item in dataset:
        logger.info(f"👉 Testing ID {item['id']}: {item['question'][:50]}...")
        
        # 1. Get Bot response
        bot_output = query_chatbot(item['question'])
        bot_answer = bot_output.get("answer", "")
        
        # 2. Get Judge scores
        judge_scores = judge_response(item, bot_answer)
        
        # 3. Store result
        res = {
            "id": item['id'],
            "question": item['question'],
            "category": item['category'],
            "language": item['language'],
            "ground_truth": item['ground_truth_answer'],
            "bot_answer": bot_answer,
            "scores": judge_scores
        }
        results.append(res)

        # 4. Aggregation logic
        cat = item['category']
        if cat not in category_metrics:
            category_metrics[cat] = {"count": 0, "completeness": 0, "grounding": 0, "correctness": 0}
        
        category_metrics[cat]["count"] += 1
        category_metrics[cat]["completeness"] += judge_scores.get("completeness", 0)
        category_metrics[cat]["grounding"] += judge_scores.get("grounding", 0)
        category_metrics[cat]["correctness"] += judge_scores.get("correctness", 0)

        # Wait to avoid rate limits
        logger.info(f"⏳ Waiting {DELAY_BETWEEN_QUESTIONS} seconds before next item...")
        time.sleep(DELAY_BETWEEN_QUESTIONS)

    # Calculate final averages
    for cat in category_metrics:
        count = category_metrics[cat]["count"]
        category_metrics[cat]["avg_completeness"] = round(category_metrics[cat]["completeness"] / count, 2)
        category_metrics[cat]["avg_grounding"] = round(category_metrics[cat]["grounding"] / count, 2)
        category_metrics[cat]["avg_correctness"] = round(category_metrics[cat]["correctness"] / count, 2)

    # Save outputs
    final_output = {
        "timestamp": datetime.now().isoformat(),
        "totals": category_metrics,
        "details": results
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    generate_markdown_report(final_output)
    logger.info(f"✅ Evaluation complete! Results saved to {OUTPUT_PATH}")

def generate_markdown_report(data: Dict):
    """Create a human-readable report."""
    with open(MARKDOWN_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# SPU Admission Chatbot - Evaluation Report\n\n")
        f.write(f"Generated on: {data['timestamp']}\n\n")
        
        f.write("## 1. Aggregate Statistics (0-2 Scale)\n\n")
        f.write("| Category | Count | Completeness | Grounding | Correctness |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        
        for cat, metrics in data['totals'].items():
            f.write(f"| {cat.capitalize()} | {metrics['count']} | {metrics['avg_completeness']} | {metrics['avg_grounding']} | {metrics['avg_correctness']} |\n")
        
        f.write("\n\n## 2. Key Failure Analysis (Scored 0)\n\n")
        failures = [d for d in data['details'] if d['scores']['correctness'] == 0]
        if not failures:
            f.write("No major correctness failures detected! 🎉\n")
        for fail in failures[:5]: # Show top 5 failures
            f.write(f"### ID {fail['id']} ({fail['category']})\n")
            f.write(f"- **Q:** {fail['question']}\n")
            f.write(f"- **Bot:** {fail['bot_answer']}\n")
            f.write(f"- **Reasoning:** {fail['scores']['reasoning']}\n\n")

if __name__ == "__main__":
    run_evaluation()
