from __future__ import annotations

import os
import re
import json
import argparse
from typing import List, Optional, Tuple, Dict, Any

from openai import OpenAI
import requests
from google import genai
from google.genai import types


def strip_think_blocks(text: str) -> str:
    if not text:
        return text
    return re.sub(r'(?is)<\s*think\s*>.*?<\s*/\s*think\s*>', '', text).strip()


def strip_numeric_citations(text: str) -> str:
    text = re.sub(r'\[(?:\s*\d+(?:\s*[-,]\s*\d+)*\s*)\]', '', text)
    text = re.sub(r'(?:\s*\[(?:\s*\d+(?:\s*[-,]\s*\d+)*\s*)\]\s*)+', ' ', text)
    return re.sub(r'\s{2,}', ' ', text).strip()


def merge_dedupe_urls(primary: List[str], secondary: List[str]) -> List[str]:
    seen, out = set(), []
    for u in primary + secondary:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def extract_urls_and_strip(text: str) -> Tuple[List[str], str]:
    """Pull URLs out of markdown/plain text, strip them from the text, and return (urls, cleaned_text)."""
    urls: List[str] = []
    DOMAIN_CORE = r'(?:[a-z0-9-]+\.)+[a-z]{2,}'
    HTTP_URL_RE = re.compile(r'https?://[^\s)]+', re.IGNORECASE)
    NAKED_DOMAIN_RE = re.compile(rf'(?<!@)\b(?:www\.)?{DOMAIN_CORE}(?:/[^\s)]+)?\b', re.IGNORECASE)

    def norm_domain_to_url(s: str) -> str:
        d = s.strip().lower()
        return d if d.startswith('http') else 'https://' + d.lstrip('www.')

    s = text

    def _paren_md_repl(m):
        urls.append(m.group(2).strip())
        return ''
    s = re.sub(r'\(\s*\[([^\]]+)\]\((https?://[^)\s]+)\)\s*\)', _paren_md_repl, s)

    s = strip_numeric_citations(s)

    def _md_repl(m):
        urls.append(m.group(2).strip())
        return m.group(1)
    s = re.sub(r'\[([^\]]+)\]\((https?://[^)\s]+)\)', _md_repl, s)

    PAREN_WITH_LINK = re.compile(
        rf'[\(（][^()\n]*?(?:https?://|(?:www\.)?{DOMAIN_CORE})[^()\n]*?[\)）]',
        re.IGNORECASE
    )

    def _paren_repl(m):
        seg = m.group(0)
        for u in HTTP_URL_RE.findall(seg):
            urls.append(u)
        for d in NAKED_DOMAIN_RE.findall(seg):
            urls.append(norm_domain_to_url(d))
        return ''
    s = PAREN_WITH_LINK.sub(_paren_repl, s)

    def _http_repl(m):
        u = m.group(0).rstrip(').,;!?:')
        urls.append(u)
        return ''
    s = HTTP_URL_RE.sub(_http_repl, s)

    def _naked_repl(m):
        d = m.group(0).rstrip(').,;!?:')
        urls.append(norm_domain_to_url(d))
        return ''
    s = NAKED_DOMAIN_RE.sub(_naked_repl, s)

    SOURCE_LINE = re.compile(r'(?im)^\s*source[s]?\s*:\s*(.+)$')

    def _source_repl(m):
        tail = m.group(1)
        for u in HTTP_URL_RE.findall(tail):
            urls.append(u)
        for d in NAKED_DOMAIN_RE.findall(tail):
            urls.append(norm_domain_to_url(d))
        return ''
    s = SOURCE_LINE.sub(_source_repl, s)

    seen = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]
    s = re.sub(r'\(\s*\)', '', s)
    s = re.sub(r'[ \t]+\n', '\n', s)
    s = re.sub(r'\n{3,}', '\n\n', s).strip()
    return urls, s


def build_answer_prompt(
    *,
    original_query: str,
    user_context: Optional[str]
) -> str:
    parts: List[str] = []
    parts.append("You are a personalized information seeking assistant.\n")

    parts += ["Original Query:", original_query, ""]

    if user_context and user_context.strip():
        parts += [
            "USER CONTEXT (follow these preferences to craft the answer):",
            user_context.strip(),
            "",
            "GUIDELINES:",
            "- Match the organization, tone, depth, and style implied by the user context.",
            "- Provide a personalized answer tailored to the user's preferences through web search.",
            "- Use accurate, up-to-date information obtained through web browsing.",
            "- Do not ask follow-up questions or provide evaluations; output only the final personalized answer.",
            ""
        ]

    parts.append("Provide a personalized answer based on user context and accurate web-searched information. Do not ask questions or provide generic responses - deliver a direct, tailored answer to the user's query.")
    return "\n".join(parts)


def call_openai_api(
    prompt: str,
    api_key: str,
    model_name: str = "gpt-4o",
    *,
    allow_search: bool = True
) -> Tuple[str, List[str]]:
    client = OpenAI(api_key=api_key or "")
    if allow_search:
        resp = client.responses.create(model=model_name, tools=[{"type": "web_search_preview"}], input=prompt)
    else:
        resp = client.responses.create(model=model_name, input=prompt)
    text = getattr(resp, "output_text", "") or ""
    return strip_numeric_citations(text), []


def call_perplexity_api(
    prompt: str,
    api_key: str,
    model_name: str = "sonar",
    *,
    allow_search: bool = True
) -> Tuple[str, List[str]]:
    if not api_key:
        raise ValueError("Missing PERPLEXITY_API_KEY")
    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "disable_search": (not allow_search),
    }
    if allow_search:
        payload["return_citations"] = True
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    choice0 = (data.get("choices") or [{}])[0]
    msg_meta = (choice0.get("message") or {}).get("metadata") or {}
    raw_cites = data.get("citations") or choice0.get("citations") or msg_meta.get("citations") or []
    urls = [u for u in raw_cites if isinstance(u, str)]
    return text.strip(), urls


def _gemini_call_blocking(prompt: str, api_key: str, model_name: str, allow_search: bool) -> Tuple[str, List[str]]:
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]) if allow_search \
        else types.GenerateContentConfig()
    response = client.models.generate_content(model=model_name, contents=prompt, config=cfg)
    data = response.to_json_dict()

    raw_text = ""
    if "text" in data and data["text"]:
        raw_text = data["text"]
    elif "candidates" in data:
        for cand in data["candidates"]:
            parts = cand.get("content", {}).get("parts", [])
            for part in parts:
                if "text" in part:
                    raw_text += part["text"] + "\n"

    clean_text = strip_numeric_citations(raw_text)

    urls: List[str] = []
    if allow_search:
        for cand in data.get("candidates", []) or []:
            gmeta = cand.get("groundingMetadata") or cand.get("grounding_metadata") or {}
            for ch in gmeta.get("groundingChunks", []) or gmeta.get("grounding_chunks", []) or []:
                web = ch.get("web") or {}
                uri = web.get("uri") or web.get("url")
                if isinstance(uri, str):
                    urls.append(uri)
            for ctx in gmeta.get("retrievedContexts", []) or gmeta.get("retrieved_contexts", []) or []:
                uri = ctx.get("uri") or ctx.get("url")
                if isinstance(uri, str):
                    urls.append(uri)
        seen = set()
        urls = [u for u in urls if not (u in seen or seen.add(u))]

    return clean_text.strip(), urls


def call_gemini_api(
    prompt: str,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    *,
    allow_search: bool = True
) -> Tuple[str, List[str]]:
    return _gemini_call_blocking(prompt, api_key, model_name, allow_search)


def call_model_api(
    prompt: str,
    *,
    model_type: str,
    model_name: str,
    openai_api_key: Optional[str] = None,
    perplexity_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    allow_search: bool = True
) -> Tuple[str, List[str]]:
    if model_type == "openai":
        return call_openai_api(prompt, api_key=openai_api_key or "", model_name=model_name, allow_search=allow_search)
    if model_type == "perplexity":
        return call_perplexity_api(prompt, api_key=perplexity_api_key or "", model_name=model_name, allow_search=allow_search)
    if model_type == "gemini":
        return call_gemini_api(prompt, api_key=gemini_api_key or "", model_name=model_name, allow_search=allow_search)
    raise ValueError(f"Unsupported model_type: {model_type}")


def run_single_inference(*, user_id: str, query: str, user_context: Optional[str], args) -> Dict[str, Any]:
    prompt = build_answer_prompt(
        user_id=user_id,
        original_query=query,
        user_context=user_context,
        enforce_search=bool(args.enforce_search),
    )

    raw_text, provider_urls = call_model_api(
        prompt,
        model_type=args.model_type,
        model_name=args.model_name,
        openai_api_key=args.openai_api_key,
        perplexity_api_key=args.perplexity_api_key,
        gemini_api_key=args.gemini_api_key,
        allow_search=bool(args.enforce_search),
    )

    urls_in_text, clean_text = extract_urls_and_strip(raw_text)
    clean_text = strip_think_blocks(clean_text)
    response_urls = merge_dedupe_urls(urls_in_text, provider_urls)

    return {
        "user_id": user_id,
        "query": query,
        "response": clean_text,
        "response_raw": raw_text,
        "response_urls": response_urls,
        "model": f"{args.model_type}:{args.model_name}",
        "source_mode": ("user_context" if user_context else "user_context_missing"),
        "user_context": (user_context or None),
    }


def main():
    p = argparse.ArgumentParser(description="Single-shot synchronous inference (personalized)")

    # Required
    p.add_argument("--user_id", type=str, required=True)
    p.add_argument("--query", type=str, required=True)

    # Models
    p.add_argument("--model_type", type=str, default="openai", choices=["openai", "perplexity", "gemini"])
    p.add_argument("--model_name", type=str, default="gpt-4o-mini", help="e.g., gpt-4o, o3, sonar, sonar-reasoning, gemini-2.0-flash, gemini-2.5-pro")
    p.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY"))
    p.add_argument("--perplexity_api_key", type=str, default=os.getenv("PERPLEXITY_API_KEY"))
    p.add_argument("--gemini_api_key", type=str, default=os.getenv("GEMINI_API_KEY"))

    # Optional user context
    p.add_argument("--user_context", type=str, default=None, help="Inline user context/preferences to personalize the answer.")

    # Browsing
    p.add_argument("--enforce_search", action="store_true", help="Allow the model to browse and ground the answer.")

    # Output
    p.add_argument("--output_path", type=str, default=None, help="If set, save a JSON result to this path.")
    p.add_argument("--print_json", action="store_true", help="Print the full JSON result to stdout instead of plain text.")

    args = p.parse_args()

    user_id = args.user_id.strip()
    query = args.query.strip()
    user_context = args.user_context.strip() if args.user_context else None

    result = run_single_inference(user_id=user_id, query=query, user_context=user_context, args=args)

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[✓] Saved: {args.output_path}")

    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result.get("response", "").strip())


if __name__ == "__main__":
    main()
