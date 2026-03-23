# 🎧 Helpdesk 2.0 — RAG + LangGraph + ChromaDB

![Python](https://img.shields.io/badge/Python-3.13-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-1.1.3-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

An intelligent helpdesk system built with **RAG (Retrieval-Augmented Generation)**, **LangGraph**, and **ChromaDB**. It automatically resolves user queries using a knowledge base, and escalates complex cases to a human agent — all from a clean Streamlit interface.

---

## 🚀 Features

- 🔍 **Advanced RAG** with MultiQueryRetriever — searches ChromaDB using multiple query variants for better results
- 🤖 **Automatic Classification** — uses DeepSeek + Pydantic structured output to decide if a query can be resolved automatically or needs escalation
- 👨‍💼 **Human-in-the-Loop** — LangGraph pauses the graph and waits for a human agent to respond when needed
- 💾 **State Persistence** — SQLite checkpointing saves graph state after each node, enabling graph resumption
- 📊 **Confidence Score** — calculates relevance score to determine response quality
- 🎫 **Ticket System** — generates unique ticket IDs and tracks all queries in real time
- 🖥️ **Streamlit Interface** — clean, responsive UI showing problems and tickets on the same screen

---

## 🏗️ Architecture

```
helpdesk-rag-graph/
│
├── src/
│   ├── config.py               # API keys and paths
│   │
│   ├── core/
│   │   └── models.py           # Pydantic models (Classification)
│   │
│   ├── rag/
│   │   ├── document_loader.py  # Load .md files with metadata
│   │   ├── text_splitter.py    # Split documents into chunks
│   │   ├── embedder.py         # OpenAI embeddings provider
│   │   ├── vector_store.py     # ChromaDB create/load
│   │   ├── retriever.py        # Simple similarity search
│   │   ├── rag_pipeline.py     # Orchestrates RAG setup
│   │   └── rag_system.py       # Advanced RAG with MultiQueryRetriever
│   │
│   ├── llm/
│   │   └── deepseek.py         # DeepSeek LLM client
│   │
│   ├── graph/
│   │   ├── state.py            # HelpdeskState definition
│   │   ├── edges.py            # Routing functions
│   │   ├── workflow.py         # Graph compilation with checkpoints
│   │   └── nodes/
│   │       ├── rag_node.py         # RAG search node
│   │       ├── classify_node.py    # Classification node
│   │       ├── escaling_node.py    # Escalation + human response node
│   │       └── response_node.py    # Final response node
│   │
│   └── ui/
│       ├── app.py              # Main Streamlit entry point
│       ├── state.py            # Session state initialization
│       └── components/
│           ├── sidebar.py          # Control panel
│           ├── problem_panel.py    # Query submission form
│           └── ticket_panel.py     # Tickets + human-in-the-loop UI
│
├── docs/                       # Knowledge base (.md files)
├── data/                       # Data directory
├── main.py                     # Development/testing entry point
├── .env                        # API keys (never commit)
├── .gitignore
└── pyproject.toml
```

---

## 🔄 System Flow

```
User submits query
        ↓
RAG Node — searches ChromaDB with MultiQueryRetriever
        ↓
Classify Node — decides: automatic or escalated?
        ↓
    automatic              escalated
        ↓                      ↓
Final Response         ⏸️ PAUSE (human-in-the-loop)
        ↓              human agent writes response
      END                      ↓
                        ▶️ RESUME graph
                               ↓
                        Final Response → END
```

---

## 🛠️ Tech Stack

| Technology            | Purpose                                   |
| --------------------- | ----------------------------------------- |
| **LangGraph**         | Agentic graph orchestration               |
| **ChromaDB**          | Vector database                           |
| **OpenAI Embeddings** | Text → vectors (`text-embedding-3-small`) |
| **DeepSeek**          | LLM for generation and classification     |
| **LangChain**         | RAG components and chains                 |
| **Streamlit**         | Web interface                             |
| **SQLite**            | Graph state persistence (checkpointing)   |
| **Pydantic**          | Structured LLM output                     |
| **UV**                | Python package manager                    |

---

## ⚙️ Installation

### Prerequisites

- Python 3.13+
- [UV](https://docs.astral.sh/uv/) package manager
- OpenAI API key
- DeepSeek API key

### Steps

**1. Clone the repository:**

```bash
git clone https://github.com/D-Amores/helpdesk-rag-graph.git
cd helpdesk-rag-graph
```

**2. Install dependencies:**

```bash
uv sync
```

**3. Create your `.env` file:**

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
```

**4. Add your knowledge base documents:**

Place your `.md` files inside the `docs/` folder:

```
docs/
├── faq.md
├── manual_usuario.md
└── guia_resolucion_problemas.md
```

**5. Run the application:**

```bash
PYTHONPATH=src uv run streamlit run src/ui/app.py
```

**6. Build the RAG knowledge base:**

Open the app and click **"Configure RAG"** in the sidebar. ✅

The system will automatically:

- Load all `.md` files from `docs/`
- Split them into chunks
- Generate embeddings with OpenAI
- Store them in ChromaDB

---

## 📖 How It Works

### RAG System (2 phases)

**Phase 1 — Setup (run once):**

```
.md files → chunks → OpenAI embeddings → ChromaDB
```

**Phase 2 — Query (every question):**

```
query → MultiQueryRetriever generates 3 variants
      → searches ChromaDB with each variant
      → DeepSeek generates response from context
```

### LangGraph State

The graph uses a typed state (`HelpdeskState`) that flows through all nodes:

```python
class HelpdeskState(TypedDict):
    consultation: str        # user query
    category: str            # automatic or escalated
    response_rag: str        # RAG response
    confidence: float        # relevance score
    sources: List[str]       # source documents
    required_human: bool     # needs human intervention
    response_human: str      # human agent response
    response_final: str      # final response to user
    history: List[str]       # accumulated processing steps
```

---

## 🧪 Testing Individual Components

```bash
# Test the full RAG pipeline
PYTHONPATH=src uv run python main.py

# Test the VectorRAGSystem
PYTHONPATH=src uv run python -c "
from rag.rag_system import VectorRAGSystem
rag = VectorRAGSystem()
result = rag.search('how do I reset my password?')
print(result)
"
```

---

## 📁 Knowledge Base Format

Place `.md` files in the `docs/` folder. The system automatically classifies them by filename:

| Filename contains | Type            |
| ----------------- | --------------- |
| `faq`             | FAQ             |
| `manual`          | Manual          |
| `troubleshooting` | Troubleshooting |
| anything else     | General         |

Example `docs/faq.md`:

```markdown
# FAQ

## How do I reset my password?

Go to the login page and click "Forgot my password"...

## How do I cancel my subscription?

Go to settings and select "Cancel subscription"...
```

---

## 🏛️ Design Principles

This project was built following:

- **SOLID** — each class has a single responsibility
- **Clean Code** — clear names, small functions, no magic numbers
- **Layered Architecture** — RAG, Graph, LLM, UI layers are fully separated
- **Dependency Injection** — components receive their dependencies, not create them

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**D-Amores**

- GitHub: [@D-Amores](https://github.com/D-Amores)

---

<div align="center">
  <small>🚀 Built with LangGraph | 🔍 ChromaDB | 💾 Checkpointing | 👨‍💼 Human-in-the-Loop</small>
</div>
