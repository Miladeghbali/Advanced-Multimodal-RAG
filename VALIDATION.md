# Validation status

## Completed in the artifact build environment
- Parsed every Python source file with Python AST successfully.
- Compiled every Python source file successfully.
- Secret scan found no real API key; only documented placeholders.
- Lightweight tests that do not require the unavailable LangChain runtime passed (citation audit, retrieval metrics, prompt-injection guardrails).

## Full integration test
The artifact build environment did not have the project LangChain/Chroma dependencies installed, and external package installation was unavailable. Therefore the complete `pytest` integration suite and model/API calls were **not falsely reported as executed here**.

After installing dependencies in a normal environment, run:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The first embedding/re-ranker run may download model files from Hugging Face. LLM and LangSmith tests additionally require the corresponding API credentials.
