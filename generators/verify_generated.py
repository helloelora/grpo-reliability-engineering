"""Verify generated questions using GPT-5.4 via OpenRouter.

Sends each question (without the answer) to GPT-5.4, compares its answer
to ours using 5% numerical tolerance. Keeps only matching questions.

Usage:
    $env:OPENROUTER_API_KEY="your-key"
    python generators/verify_generated.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "generated_variations.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "generated_verified_answer.jsonl"

VERIFIER_MODEL = "openai/gpt-5.4"
VERIFY_TEMPERATURE = 0.1
MAX_TOKENS_VERIFY = 8000
NUMERICAL_TOLERANCE = 0.05

VERIFICATION_PROMPT = """You are an expert in reliability engineering, statistics, and probability.

Solve this problem completely. Show all steps and calculations.

IMPORTANT: state your final answer as a SINGLE PLAIN NUMBER.
- No units, no commas
- Probabilities as decimals (not percentages)
- Round to 4 significant figures if needed

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


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY")
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    print(f"Verifier: {VERIFIER_MODEL}")
    print(f"Questions to verify: {len(questions)}")

    # Load already verified to support resume
    already_verified = set()
    verified = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    verified.append(item)
                    already_verified.add(item['question'][:100])
        print(f"Resuming: {len(verified)} already verified")

    passed = len(verified)
    failed = 0

    for i, item in enumerate(questions):
        if item['question'][:100] in already_verified:
            continue

        print(f"\n[{i+1}/{len(questions)}] {item['question'][:70]}...")
        print(f"  Our answer: {item['answer']}")

        try:
            resp = client.chat.completions.create(
                model=VERIFIER_MODEL,
                messages=[{"role": "user", "content": VERIFICATION_PROMPT.format(question=item['question'])}],
                temperature=VERIFY_TEMPERATURE,
                max_tokens=MAX_TOKENS_VERIFY,
            )
            ver_text = resp.choices[0].message.content
            ver_answer = extract_number(ver_text)
        except Exception as e:
            print(f"  API error: {e}")
            time.sleep(5)
            continue

        ver_std = standardize_answer(ver_answer) if ver_answer else 'N/A'
        print(f"  GPT answer: {ver_std}")

        if ver_answer and answers_match(item['answer'], ver_answer):
            print(f"  PASS")
            passed += 1
            with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            print(f"  FAIL (ours={item['answer']}, gpt={ver_std})")
            failed += 1

        time.sleep(0.5)

    print(f"\nDone. Passed: {passed}, Failed: {failed}, Rate: {passed}/{passed+failed} ({passed/(passed+failed)*100:.0f}%)")
    print(f"Verified questions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
