# LLM Mode Guide

This project now supports two runtime LLM modes in one UI:

- `local`: use the local GGUF model through `llama-cpp-python`
- `api`: use an external OpenAI-compatible chat completion API

## Default

`start_server.bat` defaults to:

```bat
LLM_PROVIDER=local
```

The frontend can still switch modes per request with the sidebar toggle.

## Local mode

Required:

```text
LLM_PROVIDER=local
MODEL_PATH=C:\path\to\EXAONE-3.5-7.8B-Instruct-Q8_0.gguf
```

## API mode

Required:

```text
API_LLM_API_KEY=your_api_key
API_LLM_ANSWER_MODEL=your_answer_model
API_LLM_ANALYSIS_MODEL=your_analysis_model
```

Optional:

```text
API_LLM_BASE_URL=https://api.openai.com/v1
API_LLM_TIMEOUT=60
LLM_PROVIDER=api
```

Notes:

- `API_LLM_BASE_URL` is for OpenAI-compatible APIs.
- If `API_LLM_ANALYSIS_MODEL` is omitted, the answer model is reused.
- The `/health` endpoint now reports both local and API readiness.
