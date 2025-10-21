# 🤖 Agentic AI Tutorial with Ollama

A hands-on tutorial for learning agentic AI patterns using **100% local, open-source models**. Build intelligent agents that can reason, retrieve information, use tools, and orchestrate complex workflows - all running on your own machine!

## 🎯 What You'll Learn

This tutorial teaches you four fundamental patterns in agentic AI:

1. **🧠 ReAct (Reasoning + Acting)** - How agents think step-by-step to solve problems
2. **📚 RAG (Retrieval Augmented Generation)** - How to ground AI responses in real data
3. **🛠️ Tool Use (MCP-style)** - How agents interact with external systems
4. **🔄 Workflow Orchestration** - How to combine everything into complex behaviors

## 📋 Prerequisites

- **Python 3.9+** 
- **8GB RAM** (16GB recommended for larger models)
- **10GB disk space** for models
- **macOS, Linux, or Windows** (with WSL2)

## 🚀 Quick Start (5 minutes)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/agentic-ai-tutorial.git
cd agentic-ai-tutorial
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv ai-agents-env
source ai-agents-env/bin/activate  # On Windows: ai-agents-env\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install and Start Ollama

**macOS:**
```bash
brew install ollama
ollama serve  # Run in a separate terminal
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
```

**Windows:**
Download from [https://ollama.ai/download](https://ollama.ai/download)

### Step 5: Pull Required Models

In a new terminal:
```bash
ollama pull llama3.2        # Main reasoning model (3GB)
ollama pull nomic-embed-text # Embedding model for RAG (274MB)
```

### Step 6: Test Your Setup

```bash
python test_ollama.py
```

You should see:
```
✅ Ollama package imported
✅ Found 2 models
✅ Generation test passed
✅ All tests passed!
```

### Step 7: Run the Application

```bash
python run.py
```

Open your browser to: **http://localhost:8501**

## 📁 Project Structure

```
agentic-ai-tutorial/
├── src/
│   ├── config.py           # Ollama configuration
│   ├── react_agent.py      # ReAct reasoning implementation
│   ├── rag_engine.py       # RAG system with vector storage
│   ├── tool_system.py      # Tool framework (MCP-style)
│   ├── workflow.py         # LangGraph workflow orchestration
│   ├── main.py            # Streamlit web interface
│   └── cli.py             # Command-line interface
├── data/                  # Data storage
│   ├── vector_store/      # FAISS vector database
│   └── traces/           # Execution traces (for debugging)
├── requirements.txt      # Python dependencies
├── run.py               # Application launcher
├── test_ollama.py       # Ollama connection test
└── README.md           # This file
```

## 🎓 Tutorial Walkthrough

### Part 1: ReAct Pattern (Reasoning + Acting)

The ReAct pattern teaches agents to think before they act.

**Try it:**
1. Go to the **ReAct tab** in the web app
2. Ask: "What is Apple's P/E ratio and is it overvalued?"
3. Watch the reasoning steps unfold

**What's happening:**
```python
# The agent follows this loop:
Thought → Action → Observation → Thought → ...
```

**Key concepts:**
- **Thought**: Agent reasons about what to do next
- **Action**: Agent selects and executes a tool
- **Observation**: Agent sees the result
- **Iteration**: Process repeats until answer found

**Code to explore:** `src/react_agent.py`

### Part 2: RAG (Retrieval Augmented Generation)

RAG prevents hallucination by grounding responses in real documents.

**Try it:**
1. Go to the **RAG tab**
2. Click "Load Financial Documents" 
3. Ask: "What are the key risks?"
4. See retrieved documents and generated answer

**What's happening:**
```python
# RAG Pipeline:
Documents → Chunk → Embed → Store → Retrieve → Generate
```

**Key concepts:**
- **Chunking**: Split documents into manageable pieces
- **Embedding**: Convert text to vectors
- **Vector Store**: FAISS database for similarity search
- **Retrieval**: Find relevant chunks for a query
- **Generation**: Create answer using retrieved context

**Code to explore:** `src/rag_engine.py`

### Part 3: Tool System (MCP-style)

Tools extend what agents can do - like APIs for AI.

**Try it:**
1. Go to the **Tools tab**
2. Select `stock_data` tool
3. Enter ticker: `AAPL`
4. Click "Execute Tool"

**Available tools:**
- **stock_data**: Fetch real market data
- **calculator**: Perform calculations
- **web_search**: Search for information (mock)

**Key concepts:**
- **Tool Schema**: Define inputs/outputs
- **Validation**: Check parameters
- **Error Handling**: Graceful failures
- **Execution History**: Track tool use

**Code to explore:** `src/tool_system.py`

### Part 4: Complete Workflow

Combines all patterns into a comprehensive analysis system.

**Try it:**
1. Go to the **Workflow tab**
2. Enter ticker: `MSFT`
3. Click "Run Complete Analysis"
4. Explore all 5 result tabs

**What you'll see:**
- **Report**: Executive summary
- **Reasoning**: Step-by-step thought process
- **RAG Context**: Retrieved documents
- **Technical**: Market indicators
- **Sentiment**: Social media analysis

**Code to explore:** `src/workflow.py`

## 🛠️ Configuration

### Environment Variables

Create a `.env` file (optional):
```bash
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434

# Model selection
REASONING_MODEL=llama3.2:latest
ANALYSIS_MODEL=llama3.2:latest
EMBEDDING_MODEL=nomic-embed-text:latest

# Settings
LOG_LEVEL=INFO
VECTOR_STORE_TYPE=faiss
```

### Using Different Models

You can experiment with different models:

```bash
# Smaller, faster model
ollama pull phi3
# Update REASONING_MODEL=phi3:latest in .env

# Better at following instructions
ollama pull mistral
# Update REASONING_MODEL=mistral:latest in .env

# Larger, more capable
ollama pull llama3.2:70b  # Requires 40GB RAM!
```

## 🔍 CLI Usage

The command-line interface is great for learning and debugging:

### Test Individual Components

```bash
# Test ReAct reasoning
python src/cli.py react "What is 2+2 and why?"

# Test RAG system
python src/cli.py rag AAPL "What are the risks?"

# Test tool execution
python src/cli.py tool stock_data --params '{"ticker": "GOOGL"}'

# Run complete workflow
python src/cli.py workflow NVDA --query "Should I invest?"
```

### Check System Status

```bash
# List available models
python src/cli.py models

# Verify setup
python src/cli.py check
```

## 🐛 Troubleshooting

### Common Issues and Solutions

| Problem | Solution |
|---------|----------|
| "Ollama not found" | Make sure `ollama serve` is running |
| "Model not found" | Run `ollama pull llama3.2` |
| No reasoning trace shown | Try `ollama pull mistral` (better at structured output) |
| Import errors | Check virtual environment is activated |
| Port 8501 in use | Run `streamlit run src/main.py --server.port 8502` |
| Out of memory | Use smaller model: `ollama pull phi3` |

### Debug Mode

Enable verbose logging:
```python
# In src/config.py
verbose: bool = True  # Set to True for debug output
```

Check console output while running for detailed logs.

## 📚 Understanding the Code

### Key Design Patterns

1. **State Management** - How data flows through the system
2. **Prompt Engineering** - Crafting effective LLM instructions  
3. **Error Recovery** - Graceful handling of failures
4. **Caching** - Preserving results across interactions
5. **Modular Design** - Each component works independently

### Learning Path

**Beginner:**
1. Run the web app and try each tab
2. Read the code comments in `src/react_agent.py`
3. Modify a prompt and see what changes

**Intermediate:**
1. Add a new tool in `src/tool_system.py`
2. Create custom documents for RAG
3. Modify the workflow to add a new step

**Advanced:**
1. Implement a new reasoning strategy
2. Add a different vector store
3. Create a production deployment

## 🎯 Educational Exercises

### Exercise 1: Add a New Tool
Create a weather tool that returns mock weather data.

**Hints:**
- Look at `StockDataTool` in `src/tool_system.py`
- Define schema with city parameter
- Return temperature and conditions

### Exercise 2: Improve RAG
Add PDF document loading to the RAG system.

**Hints:**
- Use `pypdf` library (already installed)
- Add method to `RAGEngine` class
- Handle text extraction and chunking

### Exercise 3: Custom Workflow
Create a news analysis workflow.

**Hints:**
- Combine sentiment analysis with RAG
- Add news-specific tools
- Generate summary report

## 📊 Performance Tips

- **Model Selection**: Start with smaller models for testing
- **Caching**: Results are cached in session state
- **Batch Processing**: Process multiple documents at once
- **GPU Acceleration**: Use `ollama run --gpu` if available

## 🤝 Contributing

This is an educational project! Contributions welcome:

- Add new examples
- Improve documentation
- Fix bugs
- Add new features

## 📖 Resources

### Documentation
- [Ollama Docs](https://ollama.ai/docs)
- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Streamlit Docs](https://docs.streamlit.io/)

### Key Papers
- [ReAct: Reasoning and Acting](https://arxiv.org/abs/2210.03629)
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Toolformer](https://arxiv.org/abs/2302.04761)

### Videos & Tutorials
- [Building Agents with LangGraph](https://www.youtube.com/watch?v=example)
- [Local LLMs with Ollama](https://www.youtube.com/watch?v=example)

## 📄 License

MIT License - Use freely for learning!

## 🙏 Acknowledgments

- **Ollama** team for making local LLMs accessible
- **LangChain** for the excellent framework
- **Meta, Mistral, Qwen** teams for open-source models
- The open-source AI community

---

**Made for Learning** 🎓 | **100% Local** 🏠 | **No API Keys** 🔐 | **Open Source** 📖

## Support

If you found this tutorial helpful, please ⭐ star the repository!

Questions? Open an issue on GitHub.

Happy Learning! 🚀