# Financial RAG Streamlit App

This project uses LangChain, Ollama, DuckDuckGo Search, and Streamlit to generate advanced financial reports from internal PDFs and real-time web data using Retrieval-Augmented Generation (RAG).

## Features

- Load and chunk PDF documents (e.g., annual reports)
- RAG-based Q&A using Ollama's Mistral model
- Real-time market info via DuckDuckGo web search
- Generate full financial reports using a domain-specific prompt template
- Clean Streamlit frontend for interactive queries

## App Interface

- Input your financial query (e.g., "Give insights about AMEX")
- Internally fetch answers from PDF content + web sources
- Display full report including: Executive Summary, Key Findings, Outlook, and Expert Insights

## Project Structure

```
financial-rag-streamlit/
├── app.py                    # Streamlit frontend
├── financial_rag.py          # Core logic (RAG, web search, reporting)
├── documents/                # PDF files for vector store
├── vector_db/                # Persistent Chroma DB
├── requirements.txt          # Required packages
├── setup.py                  # Editable install support
├── .gitignore
└── README.md                 # You’re here
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ajithreddy1234/financial-rag-streamlit.git
cd financial-rag-streamlit
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate         # On Windows
# OR
source venv/bin/activate      # On Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install the project in editable mode

```bash
pip install -e .
```

## Run the App

Make sure you have the Ollama model downloaded:

```bash
ollama run mistral
```

Then start the Streamlit app:

```bash
streamlit run app.py
```

## Requirements

Make sure these are installed:

```
langchain==0.2.1
langchain-community==0.2.1
chromadb==0.4.24
sentence-transformers==2.6.1
ollama==0.1.8
duckduckgo-search==5.3.0
pymupdf==1.26.3
streamlit==1.35.0
torch==2.3.1
```

If using setup.py, dependencies should still be installed via requirements.txt.

## Technologies Used

- LangChain (document loaders, RAG, PromptTemplate)
- ChromaDB (vector store)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Ollama with Mistral (local LLM)
- DuckDuckGo Search (external real-time info)
- Streamlit (UI)

## Example Query

"Generate a financial insight report for American Express including recent trends and future outlook."

Output will include:
- Executive Summary
- Key Findings
- Impact Analysis
- Future Outlook
- Expert Quotes
- Verified Sources

## Author

Ajith Reddy Pochimireddy  
GitHub: https://github.com/ajithreddy1234

## License

This project is licensed under the MIT License. Feel free to use and modify.

## Optional Features to Add

- PDF upload via UI
- Downloadable report (Markdown or PDF)
- Visual charts with Altair/Plotly
