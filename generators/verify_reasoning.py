"""Verify reasoning of answer-verified questions using GPT-5.4 via OpenRouter.

Takes questions that passed answer verification, sends the full question +
reasoning + answer to GPT-5.4, and asks it to check if the reasoning is
mathematically correct and leads to the stated answer.

Usage:
    $env:OPENROUTER_API_KEY="your-key"
    python generators/verify_reasoning.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "generated_verified_answer.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "generated_verified.jsonl"
REJECTED_PATH = PROJECT_ROOT / "data" / "generated_rejected_reasoning.jsonl"

VERIFIER_MODEL = "openai/gpt-5.4"
VERIFY_TEMPERATURE = 0.1
MAX_TOKENS_VERIFY = 4000

VERIFICATION_PROMPT = """You are an expert in reliability engineering, statistics, and probability.

I will show you a problem, a proposed step-by-step solution, and a final answer. Your task is to verify whether the reasoning is mathematically correct and logically sound.

Check for:
1. Are the formulas used correct for this type of problem?
2. Are the numerical computations accurate (allow small rounding differences)?
3. Does each step logically follow from the previous one?
4. Does the reasoning actually lead to the stated final answer?

PROBLEM:
{question}

PROPOSED REASONING:
{reasoning}

STATED ANSWER: {answer}

Respond with EXACTLY one of these two formats:
VERDICT: CORRECT
or
VERDICT: INCORRECT
Reason: [brief explanation of what is wrong]"""


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: set OPENROUTER_API_KEY")
        sys.exit(1)

    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run verify_generated.py first.")
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    questions = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line.strip()))

    print(f"Verifier: {VERIFIER_MODEL}")
    print(f"Questions to verify reasoning: {len(questions)}")

    # Support resume
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

        prompt = VERIFICATION_PROMPT.format(
            question=item['question'],
            reasoning=item['reasoning'],
            answer=item['answer'],
        )

        try:
            resp = client.chat.completions.create(
                model=VERIFIER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=VERIFY_TEMPERATURE,
                max_tokens=MAX_TOKENS_VERIFY,
            )
            ver_text = resp.choices[0].message.content
        except Exception as e:
            print(f"  API error: {e}")
            time.sleep(5)
            continue

        is_correct = "VERDICT: CORRECT" in ver_text.upper()

        if is_correct:
            print(f"  PASS")
            passed += 1
            with open(OUTPUT_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            reason = ""
            reason_match = re.search(r'Reason:\s*(.*)', ver_text, re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()[:200]
            print(f"  FAIL: {reason}")
            failed += 1
            with open(REJECTED_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps({**item, "rejection_reason": reason}, ensure_ascii=False) + '\n')

        time.sleep(0.5)

    print(f"\nDone. Passed: {passed}, Failed: {failed}")
    print(f"  Rate: {passed}/{passed+failed} ({passed/(passed+failed)*100:.0f}%)")
    print(f"  Verified: {OUTPUT_PATH}")
    if failed > 0:
        print(f"  Rejected: {REJECTED_PATH}")


if __name__ == "__main__":
    main()
