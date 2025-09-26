from pathlib import Path
import argparse
import json
import os
import sys
from evaluator import PersonalizationEvaluator


def main() -> None:
    query_id = "insert query id"
    query = "insert query here"
    user = "insert user who issued the query"
    response_text = "insert response text here",
    eval_shot_root = Path("insert path of directory containing R_J pair ex) /root/shot").resolve()
    rubric_root = Path("insert path of directory containing rubric files ex) /root/rubric").resolve()
    model = "gpt-5"
    api_key = "insert api key here"
    max_shots = 6


    if not api_key:
        print("[ERROR] OPENAI_API_KEY is not set and --api-key not provided.", file=sys.stderr)
        sys.exit(1)

    if not response_text:
        response_text = sys.stdin.read()
    response_text = str(response_text or "").strip()
    if not response_text:
        print("[ERROR] Empty response text.", file=sys.stderr)
        sys.exit(1)

    evaluator = PersonalizationEvaluator(
        eval_shot_root=Path(eval_shot_root).resolve(),
        rubric_root=Path(rubric_root).resolve(),
        model=str(model),
        api_key=api_key,
        max_shots=int(max_shots),
    )

    result = evaluator.evaluate(
        user=str(user),
        query=str(query) if query else None,
        query_id=str(query_id) if query_id else None,
        response_text=response_text,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()