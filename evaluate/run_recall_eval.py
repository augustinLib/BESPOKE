from pathlib import Path
import argparse
import json
import os
import sys
from evaluator import RecallEvaluator


def main() -> None:
    query_id = "insert query id"
    query = "insert query here"
    user = "insert user who issued the query"
    response_text = "insert response text here",
    gold_information_root = Path("Insert path of directory containing gold information files ex) /root/gold_information").resolve()
    model = "gpt-5"
    api_key = "insert api key here"


    if not api_key:
        print("[ERROR] OPENAI_API_KEY is not set and --api-key not provided.", file=sys.stderr)
        sys.exit(1)

    if not response_text:
        response_text = sys.stdin.read()
    response_text = str(response_text or "").strip()
    if not response_text:
        print("[ERROR] Empty response text.", file=sys.stderr)
        sys.exit(1)

    evaluator = RecallEvaluator(
        gold_information_root=Path(gold_information_root).resolve(),
        model=str(model),
        api_key=api_key,
    )

    result = evaluator.evaluate(
        user=str(user),
        query=str(query),
        query_id=str(query_id),
        response_text=response_text,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
