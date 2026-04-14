"""Generate variations of existing GRPO training questions.

Takes the 57 questions where the model had mixed results and generates
harder variations using Claude, verified by a second model.

Usage:
    export OPENROUTER_API_KEY="your-key"
    python generators/augment_grpo_questions.py --target 200
"""

import json
import os
import re
import random
import sys
import time
import argparse
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = PROJECT_ROOT / "data" / "dataset_grpo_combined.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "grpo_augmented.jsonl"

GENERATOR_MODEL = "anthropic/claude-opus-4.6"
VERIFIER_MODEL = "openai/gpt-5.4"
QUESTIONS_PER_BATCH = 2
NUM_SEED_EXAMPLES = 3
GEN_TEMPERATURE = 0.8
VERIFY_TEMPERATURE = 0.1
MAX_TOKENS_GEN = 16000
MAX_TOKENS_VERIFY = 8000
NUMERICAL_TOLERANCE = 0.05

GENERATION_PROMPT = """You are an expert reliability engineering professor creating exam problems.

I will show you {num_examples} example problems. Create {num_to_generate} NEW problems that test SIMILAR CONCEPTS but with DIFFERENT scenarios and parameters.

STRICT FORMAT REQUIREMENTS:
1. Each problem must be a SINGLE question with exactly ONE numeric answer
2. NO multi-part questions (no "a)", "b)", "also find", "additionally compute")
3. The question must be FULLY SELF-CONTAINED (all data included)
4. The answer must be a PLAIN NUMBER with NO UNITS and NO COMMAS
   - Probabilities as decimals: 95% -> 0.95
   - No units: just "380" not "380 hours"
   - No commas: "12000" not "12,000"

DIFFICULTY: the problem should require at least TWO of:
- Combining multiple distributions (Weibull in k-out-of-n, etc.)
- Conditional probability or Bayesian updating
- System-level analysis (series-parallel, standby, common cause)
- Parameter estimation from censored data
- Competing risks or dependent failure modes
- Accelerated life testing

# EXAMPLE PROBLEMS:

{examples}

# YOUR TASK:

Generate {num_to_generate} new problems. For each provide EXACTLY this format:

QUESTION: [Complete problem statement]

REASONING: [Complete step-by-step solution]

ANSWER: [Single plain number]

---

Generate the problems now:"""

VERIFICATION_PROMPT = """You are an expert in reliability engineering, statistics, and probability.

Solve this problem completely. Show all steps.

IMPORTANT: state your final answer as a SINGLE PLAIN NUMBER.
- No units, no commas
- Probabilities as decimals (not percentages)

Problem:
{question}

Solve step by step, then state your final numerical answer:"""


def standardize_answer(answer):
    s = str(answer).strip()
    s = re.sub(r'\\boxed\{(.*?)\}', r'\1', s)
    s = re.sub(r'\$(.*?)\$', r'\1', s)
    s = re.sub(r'(\d),(\d)', r'\1\2', s)
    m_pct = re.match(r'^([-+]?\d*\.?\d+)\s*%$', s.strip())
    if m_pct:
        s = str(round(float(m_pct.group(1)) / 100, 8))
    s = re.sub(
        r'\s*(hours?|hrs?|days?|years?|failures?|units?|FITs?)\s*$',
        '', s, flags=re.IGNORECASE)
    s = s.strip().rstrip('.')
    frac = re.match(r'^(-?\d+)/(\d+)$', s)
    if frac:
        s = str(round(int(frac.group(1)) / int(frac.group(2)), 8))
    return s


def extract_number(text):
    patterns = [
        r'[Ff]inal\s+[Aa]nswer\s*[:\-]\s*([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)',
        r'[Aa]nswer\s*[:\-]\s*([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)',
        r'=\s*([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)\s*$',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return match.group(1)
    numbers = re.findall(r'[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?', text)
    return numbers[-1] if numbers else None


def answers_match(ans1, ans2):
    a1 = standardize_answer(ans1)
    a2 = standardize_answer(ans2)
    if a1 == a2:
        return True
    try:
        v1, v2 = float(a1), float(a2)
        denom = max(abs(v1), abs(v2), 1e-10)
        return abs(v1 - v2) / denom < NUMERICAL_TOLERANCE
    except ValueError:
        return False


def parse_generated(text):
    results = []
    blocks = re.split(r'(?:^|\n)\s*(?:\*\*)?(?:#{0,3}\s*)?QUESTION\s*(?:\*\*)?:\s*', text, flags=re.IGNORECASE)
    for block in blocks[1:]:
        r_match = re.search(
            r'(?:\*\*)?(?:#{0,3}\s*)?REASONING\s*(?:\*\*)?:\s*(.*?)(?=\n\s*(?:\*\*)?(?:#{0,3}\s*)?ANSWER)',
            block, re.DOTALL | re.IGNORECASE)
        a_match = re.search(
            r'(?:\*\*)?(?:#{0,3}\s*)?ANSWER\s*(?:\*\*)?:\s*(.*?)(?:\n---|\n\n\n|\Z)',
            block, re.DOTALL | re.IGNORECASE)
        q_match = re.match(
            r'(.*?)(?=\n\s*(?:\*\*)?(?:#{0,3}\s*)?REASONING)',
            block, re.DOTALL | re.IGNORECASE)
        if not (q_match and r_match and a_match):
            continue
        question = q_match.group(1).strip()
        reasoning = r_match.group(1).strip()
        answer = a_match.group(1).strip().split('\n')[0].strip()
        if len(question) < 50 or len(reasoning) < 50:
            continue
        if re.search(r'\b[a-d]\)\s', question) or 'also find' in question.lower():
            continue
        answer = standardize_answer(answer)
        try:
            float(answer)
        except ValueError:
            continue
        results.append({
            'question': question,
            'reasoning': reasoning,
            'answer': answer,
            'answer_type': 'numeric',
            'source': 'grpo_augmented',
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=100,
                        help='Target number of verified questions')
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY")
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    print(f"Generator: {GENERATOR_MODEL}")
    print(f"Verifier:  {VERIFIER_MODEL}")
    print(f"Target:    {args.target} verified questions\n")

    with open(SEED_PATH, 'r', encoding='utf-8') as f:
        seed_data = json.load(f)
    print(f"Loaded {len(seed_data)} seed questions")

    existing_prefixes = set()
    for item in seed_data:
        existing_prefixes.add(item['question'][:200].lower())

    verified = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    verified.append(item)
                    existing_prefixes.add(item['question'][:200].lower())
        print(f"Resuming from {len(verified)} existing questions")

    batch_num = 0
    total_gen = 0
    total_checked = 0

    while len(verified) < args.target:
        batch_num += 1
        print(f"\nBatch {batch_num} | verified: {len(verified)}/{args.target}")

        examples = random.sample(seed_data, min(NUM_SEED_EXAMPLES, len(seed_data)))
        examples_text = "\n".join(
            f"Example {i+1}:\nQUESTION: {ex['question']}\n"
            f"REASONING: {ex['reasoning']}\nANSWER: {ex['answer']}\n"
            for i, ex in enumerate(examples)
        )
        prompt = GENERATION_PROMPT.format(
            num_examples=len(examples),
            num_to_generate=QUESTIONS_PER_BATCH,
            examples=examples_text,
        )

        try:
            resp = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=GEN_TEMPERATURE,
                max_tokens=MAX_TOKENS_GEN,
            )
            items = parse_generated(resp.choices[0].message.content)
        except Exception as e:
            print(f"  Generation error: {e}")
            time.sleep(5)
            continue

        # Deduplicate
        new_items = []
        for item in items:
            prefix = item['question'][:200].lower()
            if prefix not in existing_prefixes:
                existing_prefixes.add(prefix)
                new_items.append(item)

        total_gen += len(new_items)
        if not new_items:
            print("  No new items, retrying...")
            time.sleep(1)
            continue

        for item in new_items:
            total_checked += 1
            try:
                ver_resp = client.chat.completions.create(
                    model=VERIFIER_MODEL,
                    messages=[{"role": "user", "content": VERIFICATION_PROMPT.format(question=item['question'])}],
                    temperature=VERIFY_TEMPERATURE,
                    max_tokens=MAX_TOKENS_VERIFY,
                )
                ver_answer = extract_number(ver_resp.choices[0].message.content)
            except Exception as e:
                print(f"  Verification error: {e}")
                time.sleep(5)
                continue

            if ver_answer and answers_match(item['answer'], ver_answer):
                verified.append(item)
                with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                print(f"  PASS ({len(verified)}/{args.target}): {item['question'][:70]}... = {item['answer']}")
            else:
                ver_a = standardize_answer(ver_answer) if ver_answer else 'N/A'
                print(f"  FAIL: gen={item['answer']} ver={ver_a} | {item['question'][:60]}...")

            time.sleep(0.5)

        rate = len(verified) / max(total_checked, 1) * 100
        print(f"  Accept rate: {rate:.0f}%")

    print(f"\nDone. Generated: {total_gen} | Verified: {len(verified)} | Rate: {len(verified)/max(total_gen,1)*100:.0f}%")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
