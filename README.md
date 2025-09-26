# BESPOKE

<p align="center">
  <img src="./asset/github.png" alt="SPIKE Framework Overview" width="100%">
</p>


<p align="center">
  <img src="./asset/main_figure.png" alt="BESPOKE introduction" width="100%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2509.21106"><img src="https://img.shields.io/badge/📄_Paper-arXiv-red" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/yonsei-dli/BESPOKE"><img src="https://img.shields.io/badge/🤗_Hugging_Face-Dataset-yellow" alt="Hugging Face"></a>
  <a href="https://augustinlib.github.io/BESPOKE/"><img src="https://img.shields.io/badge/🌐_Website-Demo-blue" alt="Website"></a>
</p>

This is the official code for the "BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback"

## Dataset

This repository does not include datasets. All data is provided in [this repository](https://huggingface.co/datasets/yonsei-dli/BESPOKE). 

If you plan to run the evaluators, place the provided data under the `dataset/` directory so the scripts can reference it.

### Data Fields

Query Table (`queries/query_table.parquet`)

| Key            | Type | Description                                 |
| -------------- | ---- | ------------------------------------------- |
| user           | str  | User identifier                             |
| query-id       | str  | Query identifier (canonical hyphen variant) |
| query          | str  | Natural-language query                      |
| gold_information_need | str | Short description of required info      |


Gold Information (`gold_information/gold_information.parquet`)

| Key                    | Type   | Description                                                   |
| ---------------------- | ------ | ------------------------------------------------------------- |
| user                   | str    | User identifier                                               |
| query-id               | str    | Query identifier                                              |
| id                     | str    | UUID per gold info item                                      |
| gold_information_value | string | JSON/text with gold answer metadata for the query            |

Personalized Rubric (`personalized_rubric/personalized_rubric.parquet`)

| Key                      | Type   | Description                                   |
| ------------------------ | ------ | --------------------------------------------- |
| user                     | str    | User identifier                               |
| query-id                 | str    | Query identifier                              |
| id                       | str    | UUID per rubric item                          |
| personalized_rubric_value| string | Free-form or structured rubric text per query |

Meta Test (`meta_test/meta_test_set.parquet`)

| Key                | Type   | Description                                                                 |
| ------------------ | ------ | --------------------------------------------------------------------------- |
| user               | str    | User identifier                                                             |
| query-id           | str    | Query identifier                                                            |
| id                 | str    | UUID per meta test item                                                     |
| meta_test_set_value| string | JSON string with {query, gold_information_need, response_list, ...}          |

Evaluation Shot (`evaluation_shot/evaluation_shot.parquet`)

| Key                    | Type   | Description                                                    |
| ---------------------- | ------ | -------------------------------------------------------------- |
| user                   | str    | User identifier                                                |
| query-id               | str    | Query identifier                                               |
| id                     | str    | UUID per shot                                                  |
| evaluation_shot_value  | string | JSON/text few-shot example for evaluation or prompting         |

Chat/Search History (`chat_history/chat_history.parquet`, `search_history/search_history.parquet`)

| Key    | Type | Description                      |
| ------ | ---- | -------------------------------- |
| user   | str  | User identifier                  |
| id     | str  | UUID per record                  |
| history| str  | Raw text transcript blob per row |


### Data Example

Query table row (`queries/query_table.parquet`):

```
{
  "user": "user1",
  "query-id": "1",
  "query": "Why is marketing so popular these days?",
  "gold_information_need": "..."
}
```


## Evaluation

Set your API key (or pass it directly in code):

```bash
export OPENAI_API_KEY=YOUR_KEY
```

### 1) Personalization scoring

Required files:
- `dataset/evaluation_shot/{user}/query-id_{N}.json`
- `dataset/personalized_rubric/{user}/query-id_{N}.txt`

Run via the provided script (fill in placeholders inside `evaluate/run_personalizaion_eval.py`):

```bash
python evaluate/run_personalizaion_eval.py
```

Set these fields inside the script:
- `user` (e.g., "user2")
- `query_id` (e.g., 6)
- `response_text` (the response text to evaluate)
- `eval_shot_root` (e.g., `dataset/evaluation_shot`)
- `rubric_root` (e.g., `dataset/personalized_rubric`)
- `model` (default: `gpt-5`)

Output (printed as JSON):
- `need_alignment_score`, `content_depth_score`, `tone_score`, `explanation_style_score` plus corresponding feedback strings

Programmatic usage (alternative):

```python
from pathlib import Path
from evaluate.evaluator import PersonalizationEvaluator

evaluator = PersonalizationEvaluator(
    eval_shot_root=Path("dataset/evaluation_shot"),
    rubric_root=Path("dataset/personalized_rubric"),
    model="gpt-5",
)
result = evaluator.evaluate(user="user2", query_id=6, response_text="... your response ...")
```

### 2) Recall scoring

Required file:
- `dataset/gold_information/{user}/query-id_{N}.json` (format example)
```json
{
  "gold_information": ["Claim A", "Claim B"]
}
```

Run via the provided script (fill in placeholders inside `evaluate/run_recall_eval.py`):

```bash
python evaluate/run_recall_eval.py
```

Set these fields inside the script:
- `user` (e.g., "user2")
- `query_id` (e.g., 6)
- `response_text` (the response text to evaluate)
- `gold_information_root` (e.g., `dataset/gold_information`)
- `model` (default: `gpt-5`)

Output (printed as JSON):
- `matched_gold_information`: list of gold claims found in the response
- `recall`: matched fraction (0.0–1.0)

Notes:
- Paths in the scripts can be absolute or relative to the repo root.
- The scripts are templates; fill in the placeholders before running.

## Inference

CLI to generate a personalized answer for a single query. Use `--enforce_search` to enable web search/browsing for grounding.

Optional environment variables:
- OpenAI: `OPENAI_API_KEY`
- Perplexity: `PERPLEXITY_API_KEY`
- Gemini: `GEMINI_API_KEY`


Main arguments:
- `--user_id` (required), `--query` (required)
- `--user_context` Inline user preferences/context text
- `--model_type` `openai|perplexity|gemini` (default: `openai`)
- `--model_name` (default: `gpt-4o-mini`)
- `--enforce_search` Enable web search/browsing
- `--output_path` Path to save JSON result
- `--print_json` Print full JSON result

Output:
- Default: prints the final answer text only
- With `--print_json`: prints a JSON containing `response`, `response_urls`, `model`, etc.


## BibTeX

If you use this dataset, please consider citing it:

```
@misc{kim2025bespokebenchmarksearchaugmentedlarge,
      title={BESPOKE: Benchmark for Search-Augmented Large Language Model Personalization via Diagnostic Feedback}, 
      author={Hyunseo Kim and Sangam Lee and Kwangwook Seo and Dongha Lee},
      year={2025},
      eprint={2509.21106},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.21106}, 
}
```
