# Local AI Agent with RAG

A lightweight local Retrieval-Augmented Generation (RAG) application that answers pizza-related questions using restaurant reviews stored in a CSV file. The app loads review data into a vector database, retrieves the most relevant review chunks, and passes them to a local LLM running through Ollama.

## What this project does

- Reads reviews from `realistic_restaurant_reviews.csv`
- Embeds each review using `OllamaEmbeddings`
- Stores embeddings in a local Chroma vector database
- Retrieves the top relevant reviews for a user question
- Uses a local LLM (`llama3.2`) to answer the question grounded in the retrieved reviews

This is a simple example of a local AI pipeline that does not depend on cloud APIs.

## Tech stack

- Python
- LangChain
- Ollama
- Chroma
- Pandas

## Project structure

```text
.
├── main.py                     # Interactive chat loop for asking questions
├── vector.py                   # Loads data, embeds it, and creates the retriever
├── realistic_restaurant_reviews.csv
├── chrome_langchain_db/         # Persisted Chroma database (created automatically)
├── requirements.txt
├── README.md
└── .venv/                     # Optional local virtual environment
```

## Prerequisites

Before running the app, install:

- Python 3.10+
- Ollama
- A local model for the LLM and embeddings

Pull the required models once:

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the app

```bash
python main.py
```

Then enter a question such as:

- "Which pizza is best for a family?"
- "Are customers happy with the gluten-free option?"
- "What do people complain about most?"

Type `q` to quit.

## How it works

### 1. Data preparation

`vector.py` loads the CSV, builds a `Document` for each review, and stores it in Chroma.

### 2. Retrieval

When you ask a question, the retriever searches the vector database for the most relevant reviews.

### 3. Generation

`main.py` builds a prompt that includes the retrieved reviews and the user question, then sends it to the local LLM.

## Notes

- The first run may take longer because embeddings are generated and the database is created.
- The vector database is persisted in `./chrome_langchain_db`, so subsequent runs are faster.
- This project is intentionally simple and designed for local experimentation with a small, domain-specific dataset.

## Example output

```text
Ask your question (q to quit): What do customers say about the gluten-free pizza?

The reviews mention that the gluten-free option is a standout for people with celiac disease. Customers say the cauliflower crust does not fall apart, has a pleasant flavor, and is one of the best gluten-free pizzas they have tried.
```

## Troubleshooting

If Ollama is not running or the model is missing, you may see connection errors. Make sure the Ollama service is running and the required models have been pulled.

If the vector database is not created, delete the `chrome_langchain_db` folder and run the app again so it can rebuild the index.

## IPS document Q&A

The `chunking.py` module extracts and chunks PDFs in `docs/`, creates embeddings, and
saves the reusable index. `rag.py` loads that index and provides the interactive Q&A.
Run the app with:

```bash
python rag.py
```

On the first run, `rag.py` automatically builds the index with the local Ollama embedding
model `mxbai-embed-large` and stores both the text chunks and vectors in
`docs/ips_embeddings.npz`. Later runs reuse the index; it is rebuilt automatically when
the PDFs or chunking settings change. You can also run `python chunking.py` to build or
refresh the index directly. The chat model is `qwen3.5:4b`. Both models must be available
in Ollama, and scanned PDFs need OCR before indexing.

To use OpenAI for answer generation while keeping PDF embeddings and retrieval local,
run `python openai_rag.py`. Set `OPENAI_API_KEY` in the environment or in the root
`.env` file; optionally set `OPENAI_MODEL` (defaults to `gpt-4o-mini`). This still
requires Ollama for the existing `mxbai-embed-large` embeddings. Retrieved IPS excerpts
are sent to OpenAI to generate answers.
