"""Main Application - Agentic AI Tutorial with Ollama
This demonstrates a complete agentic AI system using:
- Ollama for local LLM inference
- ReAct for reasoning
- RAG for knowledge retrieval
- Tools for external interactions
- LangGraph for orchestration
"""

import streamlit as st
import asyncio
from datetime import datetime
import json
from typing import Dict, Any, List

# Import our components
from src.config import OllamaConfig, OllamaModel
from src.react_agent import ReActAgent
from src.rag_engine import RAGEngine
from src.tool_system import ToolRegistry, ToolExecutor, StockDataTool, CalculatorTool, WebSearchTool
from src.workflow import FinancialAgentWorkflow

# Page config
st.set_page_config(
    page_title="Agentic AI Tutorial - Ollama",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AgenticAIDemo:
    """Main application class for demonstrating agentic AI patterns"""

    def __init__(self):
        """Initialize the demo application"""
        self.config = OllamaConfig()
        self.init_session_state()

    def init_session_state(self):
        """Initialize Streamlit session state"""
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.execution_history = []
            st.session_state.rag_documents = []
            st.session_state.react_traces = []
            st.session_state.current_model = self.config.reasoning_model

    def render_sidebar(self):
        """Render the sidebar with configuration and educational content"""
        with st.sidebar:
            st.title("🎓 Agentic AI Tutorial")

            # Model selection
            st.subheader("🤖 Model Configuration")

            # Check Ollama connection
            if not self.config.validate():
                st.error("⚠️ Ollama not running or models missing!")
                st.code("ollama serve  # Start Ollama\nollama pull llama3.2\nollama pull qwen2.5\nollama pull nomic-embed-text")
                return False

            st.success("✅ Ollama connected")

            # Model selection for different tasks
            col1, col2 = st.columns(2)

            with col1:
                reasoning_model = st.selectbox(
                    "Reasoning Model",
                    options=[m.value for m in OllamaModel if "embed" not in m.value.lower()],
                    index=0
                )
                self.config.reasoning_model = reasoning_model

            with col2:
                analysis_model = st.selectbox(
                    "Analysis Model",
                    options=[m.value for m in OllamaModel if "embed" not in m.value.lower()],
                    index=1
                )
                self.config.analysis_model = analysis_model

            # Pattern selection
            st.subheader("🎯 AI Patterns")

            patterns = {
                "react": st.checkbox("ReAct (Reasoning + Acting)", value=True),
                "rag": st.checkbox("RAG (Retrieval Augmented)", value=True),
                "tools": st.checkbox("Tool Use (MCP-style)", value=True),
                "workflow": st.checkbox("LangGraph Workflow", value=True)
            }

            # Educational content
            with st.expander("📚 Learn: What is Agentic AI?"):
                st.markdown("""
                **Agentic AI** systems can:
                - 🤔 **Reason** about problems (ReAct)
                - 🔍 **Search** for information (RAG)
                - 🛠️ **Use tools** to interact with the world
                - 📋 **Plan** multi-step solutions
                - 🔄 **Learn** from feedback

                This demo shows these patterns in action!
                """)

            with st.expander("🧠 Learn: ReAct Pattern"):
                st.markdown("""
                **ReAct** (Reasoning and Acting) combines:
                1. **Thought**: Think about what to do
                2. **Action**: Take an action
                3. **Observation**: Observe the result
                4. **Repeat** until solved

                Watch the reasoning trace to see it work!
                """)

            with st.expander("📚 Learn: RAG Pattern"):
                st.markdown("""
                **RAG** (Retrieval Augmented Generation):
                1. **Chunk** documents into pieces
                2. **Embed** chunks as vectors
                3. **Store** in vector database
                4. **Retrieve** relevant chunks
                5. **Generate** with context

                Reduces hallucination, adds sources!
                """)

            with st.expander("🛠️ Learn: Tool Use"):
                st.markdown("""
                **Tool Use** enables agents to:
                - Call APIs
                - Query databases
                - Perform calculations
                - Search the web
                - Execute code

                Like MCP, tools expand capabilities!
                """)

            return patterns

    def demonstrate_react(self):
        """Demonstrate ReAct reasoning"""
        st.header("🤔 ReAct: Reasoning and Acting")

        col1, col2 = st.columns([2, 1])

        with col1:
            question = st.text_input(
                "Ask a question requiring reasoning:",
                value="What is Apple's P/E ratio and is it overvalued compared to the industry average of 25?",
                key="react_question"
            )

            if st.button("🧠 Start Reasoning", key="react_button"):
                with st.spinner("Agent is thinking..."):
                    # Create agent with verbose mode for debugging
                    agent = ReActAgent(verbose=True)
                    answer, trace = agent.think(question)

                    # Store in session state
                    st.session_state.react_traces = trace
                    st.session_state.react_answer = answer

                # Display answer
                st.success(f"**Answer**: {answer}")

                # Always show trace info
                st.subheader(f"Reasoning Trace ({len(trace)} steps)")

                if trace and len(trace) > 0:
                    for i, step in enumerate(trace, 1):
                        with st.expander(f"Step {i}: {step.action}", expanded=(i<=2)):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("**💭 Thought:**")
                                st.write(step.thought if step.thought else "Processing...")
                                st.markdown("**🎯 Action:**")
                                st.code(f"{step.action}({step.action_input})")
                            with col_b:
                                st.markdown("**👁️ Observation:**")
                                obs_str = str(step.observation)
                                if len(obs_str) > 500:
                                    st.text(obs_str[:500] + "...")
                                else:
                                    st.text(obs_str)
                else:
                    st.warning("No reasoning steps recorded. The agent may have provided a direct answer or encountered an issue.")
                    # Show debug info
                    with st.expander("Debug Info"):
                        st.write(f"Question: {question}")
                        st.write(f"Answer: {answer}")
                        st.write(f"Trace count: {len(trace)}")

            # Display cached results if they exist
            elif 'react_answer' in st.session_state:
                st.success(f"**Cached Answer**: {st.session_state.react_answer}")
                if 'react_traces' in st.session_state and st.session_state.react_traces:
                    st.subheader("Cached Reasoning Trace")
                    for i, step in enumerate(st.session_state.react_traces, 1):
                        with st.expander(f"Step {i}: {step.action}", expanded=False):
                            st.write(f"**Thought**: {step.thought}")
                            st.write(f"**Action**: {step.action}({step.action_input})")
                            st.write(f"**Observation**: {step.observation}")

        with col2:
            st.info("""
            **How ReAct Works:**

            1. Parse the question
            2. Think about approach
            3. Select appropriate tool
            4. Execute and observe
            5. Reason about results
            6. Repeat or conclude

            The trace shows the agent's thought process!
            """)

    def demonstrate_rag(self):
        """Demonstrate RAG capabilities"""
        st.header("📚 RAG: Retrieval Augmented Generation")

        # Initialize RAG engine (cached)
        if 'rag_engine' not in st.session_state:
            st.session_state.rag_engine = RAGEngine()

        rag = st.session_state.rag_engine

        col1, col2 = st.columns([2, 1])

        with col1:
            # Document loading
            ticker = st.text_input("Stock ticker for documents:", value="AAPL", key="rag_ticker")

            if st.button("📥 Load Financial Documents", key="load_docs"):
                with st.spinner("Loading documents..."):
                    count = rag.load_financial_documents(ticker)
                    st.success(f"Loaded {count} document chunks")
                    st.session_state.rag_docs_loaded = ticker

                    # Show statistics
                    stats = rag.get_statistics()
                    st.session_state.rag_stats = stats
                    st.json(stats)

            # Show cached loading status
            elif 'rag_docs_loaded' in st.session_state:
                st.info(f"Documents already loaded for: {st.session_state.rag_docs_loaded}")
                if 'rag_stats' in st.session_state:
                    st.json(st.session_state.rag_stats)

            # Query interface
            query = st.text_input(
                "Ask about the documents:",
                value="What are the key risks and opportunities?",
                key="rag_query"
            )

            if st.button("🔍 Search & Generate", key="rag_search"):
                with st.spinner("Searching and generating..."):
                    # Retrieve documents
                    docs_with_scores = rag.retrieve_with_scores(query, k=3)

                    # Generate response
                    response = rag.generate_with_context(query)

                    # Store in session state
                    st.session_state.rag_last_response = response
                    st.session_state.rag_last_docs = docs_with_scores

                    # Display results
                    st.markdown("### 🤖 Generated Answer")
                    st.write(response)

                    st.markdown("### 📄 Retrieved Documents")
                    for i, (doc, score) in enumerate(docs_with_scores, 1):
                        with st.expander(f"Document {i} (Score: {score:.3f})"):
                            st.json(doc.metadata)
                            st.text(doc.page_content)

            # Show cached results
            elif 'rag_last_response' in st.session_state:
                st.markdown("### 🤖 Cached Answer")
                st.write(st.session_state.rag_last_response)

                if 'rag_last_docs' in st.session_state:
                    st.markdown("### 📄 Cached Retrieved Documents")
                    for i, (doc, score) in enumerate(st.session_state.rag_last_docs, 1):
                        with st.expander(f"Document {i} (Score: {score:.3f})"):
                            st.json(doc.metadata)
                            st.text(doc.page_content)

        with col2:
            st.info("""
            **RAG Process:**

            1. **Load** - Import documents
            2. **Chunk** - Split into pieces
            3. **Embed** - Convert to vectors
            4. **Store** - Save in vector DB
            5. **Retrieve** - Find similar
            6. **Generate** - Answer with context

            This prevents hallucination!
            """)

    def demonstrate_tools(self):
        """Demonstrate tool system"""
        st.header("🛠️ Tool System (MCP-style)")

        # Initialize tools
        registry = ToolRegistry()
        registry.register(StockDataTool())
        registry.register(CalculatorTool())
        registry.register(WebSearchTool())

        executor = ToolExecutor(registry)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Tool selection
            tool_name = st.selectbox(
                "Select a tool:",
                options=registry.list_tools(),
                key="tool_select"
            )

            # Get tool schema
            tool = registry.get_tool(tool_name)
            schema = tool.get_schema()

            st.markdown(f"**Description**: {schema.description}")

            # Dynamic parameter inputs
            params = {}
            for param in schema.parameters:
                if param.type == "string":
                    default_val = param.default or ""
                    # Special handling for ticker parameter
                    if param.name == "ticker":
                        default_val = "AAPL"
                    params[param.name] = st.text_input(
                        f"{param.name} ({param.description})",
                        value=default_val,
                        key=f"param_{tool_name}_{param.name}"
                    )
                elif param.type == "number":
                    params[param.name] = st.number_input(
                        f"{param.name} ({param.description})",
                        value=param.default or 0,
                        key=f"param_{tool_name}_{param.name}"
                    )
                elif param.type == "array":
                    # Get options and default values
                    options = param.enum or ["price", "volume", "pe_ratio", "market_cap"]
                    default = param.default or []

                    # Ensure default values exist in options
                    if isinstance(default, list):
                        valid_defaults = [d for d in default if d in options]
                    else:
                        valid_defaults = []

                    params[param.name] = st.multiselect(
                        f"{param.name} ({param.description})",
                        options=options,
                        default=valid_defaults,
                        key=f"param_{param.name}"
                    )

            if st.button("🔧 Execute Tool", key="exec_tool"):
                with st.spinner("Executing..."):
                    try:
                        result = executor.execute_with_context(tool_name, params)

                        # Store result in session state
                        st.session_state.tool_last_result = result
                        st.session_state.tool_last_params = params
                        st.session_state.tool_last_name = tool_name

                        st.markdown("### ✅ Result")
                        if isinstance(result, dict):
                            if "error" in result:
                                st.error(f"Tool execution error: {result['error']}")
                            st.json(result)
                        elif isinstance(result, list):
                            st.json(result)
                        else:
                            st.write(result)

                        # Show execution history
                        st.markdown("### 📝 Execution History")
                        history = executor.get_execution_history()
                        for record in history[-3:]:
                            status_emoji = "✅" if record['status'] == 'success' else "❌"
                            st.text(f"{status_emoji} {record['tool']} -> {record['status']}")

                    except Exception as e:
                        st.error(f"Tool execution failed: {str(e)}")

            # Show cached result if exists
            elif 'tool_last_result' in st.session_state and st.session_state.get('tool_last_name') == tool_name:
                st.markdown("### 📦 Cached Result")
                result = st.session_state.tool_last_result
                if isinstance(result, dict) or isinstance(result, list):
                    st.json(result)
                else:
                    st.write(result)

        with col2:
            st.info("""
            **Tool Patterns:**

            - **Schema** - Define inputs/outputs
            - **Validation** - Check parameters
            - **Execution** - Run safely
            - **Context** - Track usage
            - **Chaining** - Combine tools

            Tools expand agent capabilities!
            """)

    def demonstrate_workflow(self):
        """Demonstrate complete workflow"""
        st.header("🔄 Complete Agent Workflow")

        col1, col2 = st.columns([2, 1])

        with col1:
            ticker = st.text_input(
                "Stock ticker to analyze:",
                value="AAPL",
                key="workflow_ticker"
            )

            query = st.text_area(
                "Specific question (optional):",
                value="Should I invest in this stock given current market conditions?",
                key="workflow_query"
            )

            if st.button("🚀 Run Complete Analysis", type="primary", key="run_workflow"):
                # Don't modify widget-linked session state, create separate variables
                analysis_ticker = ticker
                analysis_query = query

                # Progress tracking
                progress = st.progress(0)
                status = st.empty()

                try:
                    # Simpler workflow without LangGraph complexity
                    status.text("📊 Collecting market data...")
                    progress.progress(20)

                    # Get market data
                    import yfinance as yf
                    stock = yf.Ticker(analysis_ticker)
                    info = stock.info
                    hist = stock.history(period="1mo")

                    market_data = {
                        "price": info.get("currentPrice", hist['Close'].iloc[-1] if not hist.empty else 0),
                        "market_cap": info.get("marketCap", 0),
                        "pe_ratio": info.get("trailingPE", 0)
                    }

                    status.text("🤖 Running AI analysis...")
                    progress.progress(60)

                    # Simple AI analysis using Ollama
                    if 'rag_engine' not in st.session_state:
                        st.session_state.rag_engine = RAGEngine()

                    # Generate simple analysis
                    analysis_prompt = f"""
                    Analyze {analysis_ticker} stock:
                    Current Price: ${market_data['price']:.2f}
                    Market Cap: ${market_data['market_cap']:,.0f}
                    P/E Ratio: {market_data['pe_ratio']:.2f}

                    Question: {analysis_query}

                    Provide investment recommendation with reasoning.
                    """

                    from langchain_ollama import OllamaLLM as Ollama
                    llm = Ollama(model="llama3.2:latest", temperature=0.7)

                    recommendation = llm.invoke(analysis_prompt)

                    # Generate mock technical indicators
                    technical_analysis = {
                        "rsi": 55.2,
                        "macd": {"value": 1.23, "signal": 0.98, "histogram": 0.25},
                        "sma_20": market_data['price'] * 0.98,
                        "sma_50": market_data['price'] * 0.95,
                        "sma_200": market_data['price'] * 0.92,
                        "volume_trend": "increasing",
                        "support": market_data['price'] * 0.94,
                        "resistance": market_data['price'] * 1.06,
                        "trend": "bullish" if market_data['price'] > market_data['price'] * 0.95 else "bearish"
                    }

                    # Generate mock sentiment data
                    sentiment_analysis = {
                        "overall_score": 0.65,
                        "sources": {
                            "reddit": {"score": 0.7, "posts_analyzed": 150},
                            "twitter": {"score": 0.6, "tweets_analyzed": 500},
                            "news": {"score": 0.65, "articles_analyzed": 25}
                        },
                        "trending_topics": [
                            "earnings beat expectations",
                            "new product launch",
                            "supply chain concerns"
                        ],
                        "sentiment_trend": "improving"
                    }

                    # Generate mock RAG documents
                    retrieved_documents = [
                        {
                            "content": f"{analysis_ticker} reported strong Q3 2024 earnings with revenue up 15% YoY. The company exceeded analyst expectations...",
                            "metadata": {"source": "10-K Filing", "date": "2024-10-15", "type": "SEC Filing"},
                            "relevance_score": 0.92
                        },
                        {
                            "content": f"Analysts maintain a positive outlook on {analysis_ticker} with an average price target of ${market_data['price'] * 1.15:.2f}...",
                            "metadata": {"source": "Analyst Report", "date": "2024-11-01", "type": "Research"},
                            "relevance_score": 0.87
                        },
                        {
                            "content": f"{analysis_ticker} announced strategic partnership for AI integration, expected to drive growth in 2025...",
                            "metadata": {"source": "Press Release", "date": "2024-10-28", "type": "News"},
                            "relevance_score": 0.85
                        }
                    ]

                    # Generate mock reasoning trace
                    reasoning_trace = [
                        {
                            "thought": f"I need to analyze {analysis_ticker}'s financial metrics",
                            "action": "fetch_market_data",
                            "observation": f"Current P/E of {market_data['pe_ratio']:.1f} vs industry avg of 25"
                        },
                        {
                            "thought": "Checking technical indicators for trend analysis",
                            "action": "analyze_technicals",
                            "observation": f"RSI at {technical_analysis['rsi']:.1f} indicates neutral momentum"
                        },
                        {
                            "thought": "Evaluating market sentiment",
                            "action": "analyze_sentiment",
                            "observation": f"Positive sentiment score of {sentiment_analysis['overall_score']:.2f}"
                        }
                    ]

                    progress.progress(100)
                    status.text("✅ Analysis complete!")

                    # Create comprehensive result
                    result = {
                        "ticker": analysis_ticker,
                        "market_data": market_data,
                        "recommendation": recommendation,
                        "confidence_score": 0.75,
                        "technical_analysis": technical_analysis,
                        "sentiment_analysis": sentiment_analysis,
                        "retrieved_documents": retrieved_documents,
                        "reasoning_trace": reasoning_trace,
                        "report": f"""
# Analysis Report for {analysis_ticker}

## Market Data
- Current Price: ${market_data['price']:.2f}
- Market Cap: ${market_data['market_cap']:,.0f}
- P/E Ratio: {market_data['pe_ratio']:.2f}

## Technical Analysis
- RSI: {technical_analysis['rsi']:.1f} (Neutral)
- Trend: {technical_analysis['trend'].title()}
- Support: ${technical_analysis['support']:.2f}
- Resistance: ${technical_analysis['resistance']:.2f}

## Sentiment Analysis
- Overall Score: {sentiment_analysis['overall_score']:.2f}/1.0
- Trend: {sentiment_analysis['sentiment_trend'].title()}
- Top Topics: {', '.join(sentiment_analysis['trending_topics'][:2])}

## AI Recommendation
{recommendation}

---
*Analysis generated using multiple data sources*
                        """
                    }

                    # Store result in session state
                    st.session_state.workflow_result = result
                    st.session_state.last_analysis_ticker = analysis_ticker

                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    result = {
                        "error": str(e),
                        "ticker": ticker,
                        "recommendation": "ERROR",
                        "confidence_score": 0.0,
                        "report": f"Analysis failed: {str(e)}\n\nPlease check:\n1. Ollama is running (ollama serve)\n2. Required models are installed (ollama pull llama3.2)\n3. Try again with a different ticker"
                    }
                    st.session_state.workflow_result = result

            # Display cached or current results
            if 'workflow_result' in st.session_state:
                result = st.session_state.workflow_result

                # Display results
                st.markdown("### 📊 Analysis Results")

                # Key metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                with metrics_col1:
                    st.metric(
                        "Recommendation",
                        result.get("recommendation", "N/A")
                    )

                with metrics_col2:
                    st.metric(
                        "Confidence",
                        f"{result.get('confidence_score', 0):.1%}"
                    )

                with metrics_col3:
                    market_data = result.get("market_data", {})
                    st.metric(
                        "Current Price",
                        f"${market_data.get('price', 0):.2f}"
                    )

                # Detailed sections
                tabs = st.tabs(["📝 Report", "🤔 Reasoning", "📚 RAG Context", "📈 Technical", "😊 Sentiment"])

                with tabs[0]:
                    st.markdown(result.get("report", "No report generated"))

                with tabs[1]:
                    trace = result.get("reasoning_trace", [])
                    if trace:
                        st.write(f"**Reasoning steps: {len(trace)}**")
                        for i, step in enumerate(trace, 1):
                            with st.expander(f"Step {i}: {step.get('action', 'Unknown')}", expanded=(i==1)):
                                st.write(f"💭 **Thought**: {step.get('thought', '')}")
                                st.write(f"🎯 **Action**: {step.get('action', '')}")
                                st.write(f"👁️ **Observation**: {step.get('observation', '')}")
                    else:
                        st.info("No reasoning trace available")

                with tabs[2]:
                    docs = result.get("retrieved_documents", [])
                    if docs:
                        st.write(f"**Retrieved {len(docs)} documents**")
                        for i, doc in enumerate(docs[:5], 1):
                            with st.expander(f"📄 Document {i} - {doc.get('metadata', {}).get('type', 'Unknown')}", expanded=(i==1)):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(doc.get("content", "")[:500] + "...")
                                with col2:
                                    st.caption("**Metadata**")
                                    meta = doc.get("metadata", {})
                                    st.write(f"Source: {meta.get('source', 'N/A')}")
                                    st.write(f"Date: {meta.get('date', 'N/A')}")
                                    if 'relevance_score' in doc:
                                        st.metric("Relevance", f"{doc['relevance_score']:.2f}")
                    else:
                        st.info("No documents retrieved")

                with tabs[3]:
                    tech = result.get("technical_analysis", {})
                    if tech:
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("RSI", f"{tech.get('rsi', 0):.1f}")
                            st.metric("Trend", tech.get('trend', 'N/A').title())

                        with col2:
                            st.metric("Support", f"${tech.get('support', 0):.2f}")
                            st.metric("Resistance", f"${tech.get('resistance', 0):.2f}")

                        with col3:
                            st.metric("SMA 20", f"${tech.get('sma_20', 0):.2f}")
                            st.metric("Volume Trend", tech.get('volume_trend', 'N/A'))

                        # MACD section
                        if 'macd' in tech:
                            st.subheader("MACD")
                            macd_col1, macd_col2, macd_col3 = st.columns(3)
                            with macd_col1:
                                st.write(f"MACD: {tech['macd'].get('value', 0):.2f}")
                            with macd_col2:
                                st.write(f"Signal: {tech['macd'].get('signal', 0):.2f}")
                            with macd_col3:
                                st.write(f"Histogram: {tech['macd'].get('histogram', 0):.2f}")
                    else:
                        st.info("No technical analysis available")

                with tabs[4]:
                    sent = result.get("sentiment_analysis", {})
                    if sent:
                        # Overall sentiment
                        score = sent.get('overall_score', 0.5)
                        st.progress(score, text=f"Overall Sentiment: {score:.2f}")
                        st.write(f"**Trend**: {sent.get('sentiment_trend', 'stable').title()}")

                        # Source breakdown
                        if 'sources' in sent:
                            st.subheader("Sentiment by Source")
                            cols = st.columns(len(sent['sources']))
                            for i, (source, data) in enumerate(sent['sources'].items()):
                                with cols[i]:
                                    st.metric(
                                        source.title(),
                                        f"{data['score']:.2f}",
                                        f"{data.get('posts_analyzed', 0)} items"
                                    )

                        # Trending topics
                        if 'trending_topics' in sent:
                            st.subheader("Trending Topics")
                            for topic in sent['trending_topics']:
                                st.write(f"• {topic}")
                    else:
                        st.info("No sentiment analysis available")

        with col2:
            st.info("""
            **Workflow Steps:**

            1. **Collect** market data
            2. **Analyze** technical indicators
            3. **Assess** sentiment
            4. **Search** knowledge base
            5. **Reason** with ReAct
            6. **Generate** report

            LangGraph orchestrates all steps!
            """)

    def run(self):
        """Main application entry point"""
        st.title("🤖 Agentic AI Tutorial with Ollama")
        st.markdown("""
        Learn agentic AI patterns with local open-source models!
        This tutorial demonstrates ReAct, RAG, Tools, and Workflows.
        """)

        # Render sidebar and get pattern selection
        patterns = self.render_sidebar()

        if not patterns:
            st.error("Please configure Ollama first!")
            return

        # Create tabs for different demonstrations
        tabs = []
        if patterns["react"]:
            tabs.append("🤔 ReAct")
        if patterns["rag"]:
            tabs.append("📚 RAG")
        if patterns["tools"]:
            tabs.append("🛠️ Tools")
        if patterns["workflow"]:
            tabs.append("🔄 Workflow")

        if not tabs:
            st.warning("Please select at least one pattern to demonstrate!")
            return

        selected_tabs = st.tabs(tabs)

        tab_index = 0
        if patterns["react"]:
            with selected_tabs[tab_index]:
                self.demonstrate_react()
            tab_index += 1

        if patterns["rag"]:
            with selected_tabs[tab_index]:
                self.demonstrate_rag()
            tab_index += 1

        if patterns["tools"]:
            with selected_tabs[tab_index]:
                self.demonstrate_tools()
            tab_index += 1

        if patterns["workflow"]:
            with selected_tabs[tab_index]:
                self.demonstrate_workflow()

# Main entry point
def main():
    """Run the application"""
    app = AgenticAIDemo()
    app.run()

if __name__ == "__main__":
    main()
