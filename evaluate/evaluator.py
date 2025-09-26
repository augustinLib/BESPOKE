from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import OpenAI


def read_frame(input_path: Path):
    try:
        import pandas as pd  
    except Exception as exc: 
        raise RuntimeError(f"pandas is required: {exc}")

    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        engine: Optional[str] = None
        try:
            import pyarrow as _ 

            engine = "pyarrow"
        except Exception:
            try:
                import fastparquet as _  

                engine = "fastparquet"
            except Exception:
                engine = None
        if engine is None:
            raise RuntimeError(
                "Neither pyarrow nor fastparquet available to read Parquet. Provide CSV instead."
            )
        return pd.read_parquet(input_path, engine=engine)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    raise ValueError(f"Unsupported input extension: {suffix}")


def write_outputs(df, output_parquet: Path, output_csv: Optional[Path] = None) -> None:
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    engine: Optional[str] = None
    try:
        import pyarrow as _  
        engine = "pyarrow"
    except Exception:
        try:
            import fastparquet as _  

            engine = "fastparquet"
        except Exception:
            engine = None

    if engine is not None:
        try:
            df.to_parquet(output_parquet, index=False, engine=engine)
            print(f"Wrote Parquet: {output_parquet} ({len(df):,} rows)")
            return
        except Exception as exc:  
            print(f"[WARN] Parquet write failed via '{engine}': {exc}")

    csv_path = output_csv or output_parquet.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote CSV fallback: {csv_path} ({len(df):,} rows)")



@dataclass
class ShotExample:
    contents: str
    need_alignment_score: Any
    need_alignment_feedback: str
    content_depth_score: Any
    content_depth_feedback: str
    tone_score: Any
    tone_feedback: str
    explanation_style_score: Any
    explanation_style_feedback: str


@dataclass
class ShotBundle:
    query: str
    gold_information_need: str
    examples: List[ShotExample]
    rubric_text: str = ""


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def slugify_preserve_case(text: str) -> str:
    """Create a filename-friendly slug but preserve original casing.

    - Replace non-alphanumeric with underscores
    - Collapse repeated underscores
    - Trim leading/trailing underscores
    """
    s = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def try_load_shot_file(file_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _normalize_query_id(raw: Any) -> str:
    s = str(raw).strip()
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else s


def find_eval_shot_json_by_query_id(user: str, query_id: Any, eval_shot_root: Path) -> Optional[Dict[str, Any]]:
    user_dir = eval_shot_root / user
    if not user_dir.exists():
        return None
    qid = _normalize_query_id(query_id)
    candidate = user_dir / f"query-id_{qid}.json"
    return try_load_shot_file(candidate)


def parse_shot_bundle(shot_json: Dict[str, Any], max_shots: int) -> ShotBundle:
    query: str = str(shot_json.get("query", ""))
    gin: str = str(shot_json.get("gold_information_need", ""))
    response_list: List[Dict[str, Any]] = shot_json.get("response_list", []) or []

    examples: List[ShotExample] = []
    for response in response_list[:max_shots]:
        examples.append(
            ShotExample(
                contents=str(response.get("contents", "")),
                need_alignment_score=(response.get("need_alignment", {}) or {}).get("score"),
                need_alignment_feedback=str((response.get("need_alignment", {}) or {}).get("feedback", "")),
                content_depth_score=(response.get("content_depth", {}) or {}).get("score"),
                content_depth_feedback=str((response.get("content_depth", {}) or {}).get("feedback", "")),
                tone_score=(response.get("tone", {}) or {}).get("score"),
                tone_feedback=str((response.get("tone", {}) or {}).get("feedback", "")),
                explanation_style_score=(response.get("explanation_style", {}) or {}).get("score"),
                explanation_style_feedback=str((response.get("explanation_style", {}) or {}).get("feedback", "")),
            )
        )

    return ShotBundle(query=query, gold_information_need=gin, examples=examples)


def find_personalized_rubric_text_by_query_id(user: str, query_id: Any, rubric_root: Path) -> str:
    user_dir = rubric_root / user
    if not user_dir.exists():
        raise FileNotFoundError(f"Rubric dir not found for user='{user}': {user_dir}")
    qid = _normalize_query_id(query_id)
    p = user_dir / f"query-id_{qid}.txt"
    if not p.exists():
        raise FileNotFoundError(
            f"Personalized rubric not found for user='{user}', query-id='{qid}' under {user_dir}"
        )
    return p.read_text(encoding="utf-8")


def parse_json_safely(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3]
    return json.loads(cleaned)


def parse_tagged_text_output(text: str) -> Dict[str, Any]:
    def strip_fences(t: str) -> str:
        cleaned_t = t.strip()
        if cleaned_t.startswith("```") and cleaned_t.endswith("```"):
            cleaned_t = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned_t).strip()
            cleaned_t = cleaned_t[:-3] if cleaned_t.endswith("```") else cleaned_t
        return cleaned_t

    s = strip_fences(text)

    def find_score(label: str) -> Optional[int]:
        m = re.search(rf"(?:<\s*{label}\s*>\s*:|\b{label}\b\s*:)\s*([1-5])\b", s, flags=re.IGNORECASE)
        if not m:
            return None
        return coerce_score(m.group(1))

    def find_feedback(label: str) -> str:
        next_tag = r"\n(?:<[^>]+>\s*:|(?:Need Alignment|Content Depth|Tone|Explanation Style)\s(?:Score|Feedback)\s*:)"
        pattern = rf"(?:<\s*{label}\s*>\s*:|\b{label}\b\s*:)\s*(.*?)(?={next_tag}|\Z)"
        m = re.search(pattern, s, flags=re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    return {
        "need_alignment_score": find_score("Need Alignment Score"),
        "need_alignment_feedback": find_feedback("Need Alignment Feedback"),
        "content_depth_score": find_score("Content Depth Score"),
        "content_depth_feedback": find_feedback("Content Depth Feedback"),
        "tone_score": find_score("Tone Score"),
        "tone_feedback": find_feedback("Tone Feedback"),
        "explanation_style_score": find_score("Explanation Style Score"),
        "explanation_style_feedback": find_feedback("Explanation Style Feedback"),
    }



def coerce_score(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(str(value).strip())
    except Exception:
        return None


SYNC_INPUT = Union[List[Dict[str, str]], str]


class PersonalizationEvaluator:

    def __init__(
        self,
        *,
        eval_shot_root: Path,
        rubric_root: Path,
        model: str = "gpt-5",
        api_key: Optional[str] = None,
        max_shots: int = 5,
    ) -> None:
        self.eval_shot_root = eval_shot_root
        self.rubric_root = rubric_root
        self.model = str(model)
        self.max_shots = int(max_shots)
        self.client = OpenAI(api_key=(api_key or os.getenv("OPENAI_API_KEY") or None))


    def _messages_to_responses_input(self, messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
        role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
        for m in messages:
            role = role_map.get(m.get("role", "user"), "User")
            content = str(m.get("content", ""))
            parts.append(f"<{role}>:\n{content}")
        return "\n\n".join(parts)

    def _extract_text_from_responses(self, resp: Any) -> str:
        # Try the convenience property first
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text
        # Fallback: try to navigate common structures
        try:
            output = getattr(resp, "output", None) or []
            chunks: List[str] = []
            for item in output:
                content = getattr(item, "content", None) or []
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str) and t:
                        chunks.append(t)
            if chunks:
                return "\n".join(chunks).strip()
        except Exception:
            pass
        try:
            choices = getattr(resp, "choices", None) or []
            if choices:
                msg = getattr(choices[0], "message", None)
                if msg is not None:
                    return getattr(msg, "content", "") or ""
        except Exception:
            pass
        # Give up
        return ""


    def _parse_tagged_text_output(self, text: str) -> Dict[str, Any]:
        def strip_fences(t: str) -> str:
            cleaned_t = t.strip()
            if cleaned_t.startswith("```") and cleaned_t.endswith("```"):
                cleaned_t = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned_t).strip()
                cleaned_t = cleaned_t[:-3] if cleaned_t.endswith("```") else cleaned_t
            return cleaned_t
    
        s = strip_fences(text)
    
        def find_score(label: str) -> Optional[int]:
            m = re.search(rf"(?:<\s*{label}\s*>\s*:|\b{label}\b\s*:)\s*([1-5])\b", s, flags=re.IGNORECASE)
            if not m:
                return None
            return coerce_score(m.group(1))
    
        def find_feedback(label: str) -> str:
            next_tag = r"\n(?:<[^>]+>\s*:|(?:Need Alignment|Content Depth|Tone|Explanation Style)\s(?:Score|Feedback)\s*:)"
            pattern = rf"(?:<\s*{label}\s*>\s*:|\b{label}\b\s*:)\s*(.*?)(?={next_tag}|\Z)"
            m = re.search(pattern, s, flags=re.IGNORECASE | re.DOTALL)
            return m.group(1).strip() if m else ""
    
        return {
            "need_alignment_score": find_score("Need Alignment Score"),
            "need_alignment_feedback": find_feedback("Need Alignment Feedback"),
            "content_depth_score": find_score("Content Depth Score"),
            "content_depth_feedback": find_feedback("Content Depth Feedback"),
            "tone_score": find_score("Tone Score"),
            "tone_feedback": find_feedback("Tone Feedback"),
            "explanation_style_score": find_score("Explanation Style Score"),
            "explanation_style_feedback": find_feedback("Explanation Style Feedback"),
        }
    

    def _parse_shot_bundle(self, shot_json: Dict[str, Any], max_shots: int) -> ShotBundle:
        query: str = str(shot_json.get("query", ""))
        gin: str = str(shot_json.get("gold_information_need", ""))
        response_list: List[Dict[str, Any]] = shot_json.get("response_list", []) or []

        examples: List[ShotExample] = []
        for response in response_list[:max_shots]:
            examples.append(
                ShotExample(
                    contents=str(response.get("contents", "")),
                    need_alignment_score=(response.get("need_alignment", {}) or {}).get("score"),
                    need_alignment_feedback=str((response.get("need_alignment", {}) or {}).get("feedback", "")),
                    content_depth_score=(response.get("content_depth", {}) or {}).get("score"),
                    content_depth_feedback=str((response.get("content_depth", {}) or {}).get("feedback", "")),
                    tone_score=(response.get("tone", {}) or {}).get("score"),
                    tone_feedback=str((response.get("tone", {}) or {}).get("feedback", "")),
                    explanation_style_score=(response.get("explanation_style", {}) or {}).get("score"),
                    explanation_style_feedback=str((response.get("explanation_style", {}) or {}).get("feedback", "")),
                )
            )

        return ShotBundle(query=query, gold_information_need=gin, examples=examples)

    def _find_eval_shot_json(self, user: str, query: str) -> Optional[Dict[str, Any]]:
        user_dir = self.eval_shot_root / user
        if not user_dir.exists():
            return None

        for candidate in sorted(user_dir.glob("*.json")):
            data = try_load_shot_file(candidate)
            if not data:
                continue
            if str(data.get("query", "")) == query:
                return data

        slug = slugify(query)
        fallback = user_dir / f"{user}-{slug}.json"
        if fallback.exists():
            return try_load_shot_file(fallback)
        return None


    def _find_eval_shot_json_by_query_id(self, user: str, query_id: Any) -> Optional[Dict[str, Any]]:
        user_dir = self.eval_shot_root / user
        if not user_dir.exists():
            return None 
        qid = _normalize_query_id(query_id)
        candidate = user_dir / f"query-id_{qid}.json"
        return try_load_shot_file(candidate)


    def _find_personalized_rubric_text(self, user: str, query: str) -> str:
        """Find and return the personalized rubric text for a given user and query.

        Tries multiple filename strategies under {rubric_root}/{user}:
          - Exact: {user}-{SlugPreserveCase(query)}_criteria.txt
          - Lowercase slug: {user}-{slugify(query)}_criteria.txt
          - Case-insensitive or punctuation-insensitive matches against files ending with _criteria.txt
        """
        user_dir = self.rubric_root / user
        if not user_dir.exists():
            raise FileNotFoundError(f"Rubric dir not found for user='{user}': {user_dir}")

        cand_a = f"{user}-{slugify_preserve_case(query)}_criteria.txt"
        cand_b = f"{user}-{slugify(query)}_criteria.txt"

        def try_read(p: Path) -> Optional[str]:
            if p.exists():
                return p.read_text(encoding="utf-8")
            return None

        # Direct exact matches
        for name in (cand_a, cand_b):
            txt = try_read(user_dir / name)
            if txt is not None:
                return txt

        # Case-insensitive filename match
        target_names_lower = {cand_a.lower(), cand_b.lower()}
        for p in user_dir.glob("*_criteria.txt"):
            if p.name.lower() in target_names_lower:
                txt = try_read(p)
                if txt is not None:
                    return txt


    def _find_personalized_rubric_text_by_query_id(self, user: str, query_id: Any) -> str:
        user_dir = self.rubric_root / user
        if not user_dir.exists():
            raise FileNotFoundError(f"Rubric dir not found for user='{user}': {user_dir}")
        qid = _normalize_query_id(query_id)
        p = user_dir / f"query-id_{qid}.txt"
        if not p.exists():
            raise FileNotFoundError(
                f"Personalized rubric not found for user='{user}', query-id='{qid}' under {user_dir}"
            )
        return p.read_text(encoding="utf-8")


    def _build_messages_for_eval(self, bundle: ShotBundle, new_response_text: str) -> SYNC_INPUT:
        detailed_instruction = f"""
        "You are evaluating responses exactly like the specific human who wrote the examples.
        Replicate their preferences, strictness/leniency, tone, and feedback style precisely, including average score levels and feedback length from the examples.
        First, recall the gold_information_need as the user's underlying intent, and use it as the primary reference for all judgments.
        Use the rubric below for scoring each criterion from 1 to 5 (Only integers are allowed).
        For each criterion, think step-by-step: (1) Identify key elements from the response, (2) Compare to examples and gold_information_need, (3) Assign score based on rubric, (4) Provide concise feedback mirroring example style.

        Personalized rubric for this user and query:
        {bundle.rubric_text}

        Instructions:
        - Personalize judgments to match the examples exactly; if patterns show leniency or strictness on any criterion (e.g., tone or Need Alignment), apply similarly across all evaluations while referencing average scores from examples.
        - Be concise and actionable in feedback. Mirror the example evaluator's language, politeness level, and any emojis precisely.
        - First, think step-by-step for each criterion between <THINK> and </THINK> tags, write your thoughts.
        - Then, provide the score and feedback.


        <USER_INPUT>
        Query: {bundle.query}
        Gold Information Need: {bundle.gold_information_need}
        <END_USER_INPUT>


        <EXAMPLES>
        """

        for example in bundle.examples:
            example_assistant_text = f"""
            Response: {example.contents}
            Need Alignment Score: {example.need_alignment_score}
            Need Alignment Feedback: {example.need_alignment_feedback}
            Content Depth Score: {example.content_depth_score}
            Content Depth Feedback: {example.content_depth_feedback}
            Tone Score: {example.tone_score}
            Tone Feedback: {example.tone_feedback}
            Explanation Style Score: {example.explanation_style_score}
            Explanation Style Feedback: {example.explanation_style_feedback}


            """
            detailed_instruction += example_assistant_text

        evaluate_user_input = f"""
        <END_EXAMPLES>

        <EVALUATE_USER_INPUT>
        Response: {new_response_text}

        Output strictly with the following fields (integers 1-5 for scores):
        Need Alignment Score: [1-5]
        Need Alignment Feedback: ...
        Content Depth Score: [1-5]
        Content Depth Feedback: ...
        Tone Score: [1-5]
        Tone Feedback: ...
        Explanation Style Score: [1-5]
        Explanation Style Feedback: ...

        <THINK>
        """

        detailed_instruction += evaluate_user_input
        return detailed_instruction


    def evaluate(
        self,
        *,
        user: str,
        response_text: str,
        query: Optional[str] = None,
        query_id: Optional[Union[str, int]] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        if not (query or query_id):
            raise ValueError("Must provide either query or query_id")

        # Load shot JSON and rubric per call
        if query_id is not None:
            shot_json = self._find_eval_shot_json_by_query_id(user, query_id)
            if not shot_json:
                raise FileNotFoundError(
                    f"No evaluation_shot JSON found for user='{user}' query-id='{query_id}' under {self.eval_shot_root / user}"
                )
            bundle = self._parse_shot_bundle(shot_json, max_shots=self.max_shots)
            rubric_text = self._find_personalized_rubric_text_by_query_id(user, query_id)
        else:
            assert query is not None
            shot_json = self._find_eval_shot_json(user, query)
            if not shot_json:
                raise FileNotFoundError(
                    f"No evaluation_shot JSON found for user='{user}' query='{query}' under {self.eval_shot_root / user}"
                )
            bundle = self._parse_shot_bundle(shot_json, max_shots=self.max_shots)
            rubric_text = self._find_personalized_rubric_text(user, query)

        bundle.rubric_text = rubric_text

        messages_or_prompt: SYNC_INPUT = self._build_messages_for_eval(bundle, response_text)
        if isinstance(messages_or_prompt, list):
            responses_input = self._messages_to_responses_input(messages_or_prompt)
        else:
            responses_input = messages_or_prompt

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                completion = self.client.responses.create(
                    model=self.model,
                    input=responses_input,
                    reasoning={"effort": "high"},
                )
                text = self._extract_text_from_responses(completion)
                parsed = self._parse_tagged_text_output(text)
                return {
                    "need_alignment_score": parsed.get("need_alignment_score"),
                    "need_alignment_feedback": parsed.get("need_alignment_feedback", ""),
                    "content_depth_score": parsed.get("content_depth_score"),
                    "content_depth_feedback": parsed.get("content_depth_feedback", ""),
                    "tone_score": parsed.get("tone_score"),
                    "tone_feedback": parsed.get("tone_feedback", ""),
                    "explanation_style_score": parsed.get("explanation_style_score"),
                    "explanation_style_feedback": parsed.get("explanation_style_feedback", ""),
                }
            except Exception as exc:  
                last_error = exc
                sleep_seconds = 1.5 * attempt
                print(
                    f"[WARN] Eval attempt {attempt} failed: {exc}. Retrying in {sleep_seconds:.1f}s…",
                    flush=True,
                )
                time.sleep(sleep_seconds)

        raise RuntimeError(f"Evaluation failed after {max_retries} attempts: {last_error}")



class RecallEvaluator:
    def __init__(
        self,
        *,
        gold_information_root: Path,
        model: str = "gpt-5",
        api_key: Optional[str] = None,
    ) -> None:
        self.gold_information_root = gold_information_root
        self.model = str(model)
        self.client = OpenAI(api_key=(api_key or os.getenv("OPENAI_API_KEY") or None))


    def _try_load_gold_information_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None 

    def _find_gold_information_json_by_query_id(self, user: str, query_id: Any) -> Optional[Dict[str, Any]]:
        user_dir = self.gold_information_root / user
        if not user_dir.exists():
            return None
        qid = _normalize_query_id(query_id)
        candidate = user_dir / f"query-id_{qid}.json"
        return self._try_load_gold_information_file(candidate)

    def _extract_claims(self, gold_data: Optional[Dict[str, Any]]) -> List[str]:
        if not isinstance(gold_data, dict):
            return []
        claims = gold_data.get("gold_information", [])
        return [str(c) for c in claims if isinstance(c, (str, int, float))]

    def _is_claim_in_response(self, *, response_text: str, claim: str, max_retries: int = 3) -> bool:
        system_msg = (
            "You are a precise fact checker. Decide if the response contains the claim's core meaning. "
            "Answer strictly with true or false. Consider it contained if semantically equivalent even with different words. "
            "Do not count if contradicted or absent."
        )
        user_msg = (
            "Claim: "
            + claim
            + "\n\nResponse:\n"
            + response_text
            + "\n\nReturn strictly one token: true or false."
        )

        for attempt in range(1, max_retries + 1):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
                content = (completion.choices[0].message.content or "").strip().lower()
                if content.startswith("true") or content.startswith("yes") or content.startswith("1"):
                    return True
                if content.startswith("false") or content.startswith("no") or content.startswith("0"):
                    return False
            except Exception as exc:
                if attempt == max_retries:
                    print(f"Error in claim check (final): {exc}")
                else:
                    time.sleep(1.5 * attempt)
        return False

    def evaluate(
        self,
        *,
        user: str,
        response_text: str,
        query: Optional[str] = None,
        query_id: Optional[Union[str, int]] = None,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        if not (query or query_id):
            raise ValueError("Must provide either query or query_id")

        if query_id is not None:
            gold_json = self._find_gold_information_json_by_query_id(user, query_id)
            if not gold_json:
                raise FileNotFoundError(
                    f"No gold_information JSON found for user='{user}' query-id='{query_id}' under {self.gold_information_root / user}"
                )
        else:
            assert query is not None
            gold_json = self._find_gold_information_json(user, query)
            if not gold_json:
                raise FileNotFoundError(
                    f"No gold_information JSON found for user='{user}' query='{query}' under {self.gold_information_root / user}"
                )

        claims: List[str] = self._extract_claims(gold_json)
        if not claims:
            return {"matched_gold_information": [], "recall": 0.0}

        included_claims: List[str] = []
        for claim in claims:
            if self._is_claim_in_response(response_text=response_text, claim=claim, max_retries=max_retries):
                included_claims.append(claim)

        denom = len(claims)
        recall_value = (len(included_claims) / denom) if denom > 0 else 0.0
        return {
            "matched_gold_information": included_claims,
            "recall": recall_value,
        }