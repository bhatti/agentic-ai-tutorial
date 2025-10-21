"""RAG (Retrieval-Augmented Generation) Engine
This demonstrates key RAG concepts:
- Document loading and chunking
- Vector embeddings with local models
- Similarity search
- Context-aware generation
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os
import json
from pathlib import Path

from langchain_ollama import OllamaLLM as Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

@dataclass
class RAGDocument:
    """Represents a document in the RAG system"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class RAGEngine:
    """RAG Engine for document retrieval and augmented generation

    This implementation demonstrates:
    - Multiple document source handling
    - Chunking strategies
    - Vector store operations
    - Retrieval strategies
    - Augmented generation
    """

    def __init__(self,
                 embedding_model: str = "nomic-embed-text:latest",
                 llm_model: str = "llama3.2:latest",
                 vector_store_type: str = "chroma",
                 persist_directory: str = "./data/vector_store",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50):
        """Initialize RAG engine

        Args:
            embedding_model: Ollama model for embeddings
            llm_model: Ollama model for generation
            vector_store_type: Type of vector store (chroma/faiss)
            persist_directory: Directory to persist vector store
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        try:
            from src.config import config
        except ModuleNotFoundError:
            from config import config

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=embedding_model
        )

        # Initialize LLM
        self.llm = Ollama(
            model=llm_model,
            temperature=0.7
        )

        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Initialize vector store
        self.vector_store_type = vector_store_type
        self.persist_directory = persist_directory
        self.vector_store = self._initialize_vector_store()

        # Document sources registry
        self.document_sources = {
            "financial_reports": [],
            "news_articles": [],
            "market_analysis": [],
            "company_filings": []
        }

    def _initialize_vector_store(self):
        """Initialize or load vector store"""
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Use FAISS for Mac compatibility
        import faiss
        from langchain_community.vectorstores import FAISS

        index_path = os.path.join(self.persist_directory, "faiss_index")
        if os.path.exists(index_path):
            try:
                return FAISS.load_local(index_path, self.embeddings, allow_dangerous_deserialization=True)
            except:
                # Create new if loading fails
                return FAISS.from_texts(["init"], self.embeddings)
        else:
            # Create empty FAISS index
            return FAISS.from_texts(["init"], self.embeddings)

    def add_documents(self, documents: List[Dict[str, Any]], source_type: str = "general"):
        """Add documents to the RAG system

        Args:
            documents: List of documents with 'content' and 'metadata'
            source_type: Type of document source
        """
        # Convert to Document objects
        docs = []
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            metadata["source_type"] = source_type

            # Split into chunks
            chunks = self.text_splitter.split_text(content)

            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_id"] = i
                chunk_metadata["total_chunks"] = len(chunks)

                docs.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))

        # Add to vector store
        self.vector_store.add_documents(docs)

        # Track source
        self.document_sources[source_type].extend(docs)

        # Persist FAISS index
        if self.persist_directory:
            index_path = os.path.join(self.persist_directory, "faiss_index")
            self.vector_store.save_local(index_path)

        return len(docs)

    def load_financial_documents(self, ticker: str = "AAPL"):
        """Load financial documents for a company (demonstration)"""

        # In a real implementation, this would fetch actual documents
        # Here we create sample documents for demonstration

        sample_docs = [
            {
                "content": f"""
                {ticker} Financial Report Q3 2024

                Revenue: $94.9 billion, up 6% year over year
                Net Income: $22.9 billion
                Earnings Per Share: $1.46

                The company showed strong performance across all product categories.
                iPhone revenue grew 5% with strong demand for the iPhone 15 Pro models.
                Services revenue reached an all-time high of $22.3 billion.

                Gross margin was 45.2%, reflecting the strength of our ecosystem.
                Operating cash flow was $28.7 billion.

                Outlook: We expect continued growth in the December quarter driven by
                holiday sales and strong services adoption.
                """,
                "metadata": {
                    "ticker": ticker,
                    "document_type": "earnings_report",
                    "quarter": "Q3",
                    "year": 2024,
                    "source": "company_filing"
                }
            },
            {
                "content": f"""
                Market Analysis: {ticker} Stock Assessment

                Technical Analysis:
                - Current RSI: 58 (neutral territory)
                - 50-day MA: $175, 200-day MA: $165
                - Support level: $170, Resistance: $185
                - MACD showing bullish crossover

                Fundamental Analysis:
                - P/E Ratio: 29.5 (slightly above industry average)
                - PEG Ratio: 2.8
                - Dividend Yield: 0.44%
                - Return on Equity: 147%

                Analyst Consensus:
                - 24 Buy ratings, 8 Hold, 1 Sell
                - Average price target: $195

                Key Risks:
                - Regulatory scrutiny in EU and US
                - China market exposure
                - Smartphone market saturation

                Key Opportunities:
                - AI integration across products
                - Vision Pro ecosystem development
                - Services segment expansion
                """,
                "metadata": {
                    "ticker": ticker,
                    "document_type": "analyst_report",
                    "date": "2024-10-15",
                    "source": "market_research"
                }
            },
            {
                "content": f"""
                Recent News: {ticker} Announces AI Partnership

                The company announced a strategic partnership with a leading AI research lab
                to integrate advanced language models into its operating systems.

                This move is expected to enhance Siri capabilities and introduce new
                AI-powered features across the product lineup. The integration will
                begin with iOS 18.2 and macOS 15.1.

                Market reaction was positive with shares up 2.3% on the news.
                Analysts view this as a critical step in maintaining competitive edge
                against rivals who have already integrated similar AI capabilities.

                The partnership includes provisions for on-device processing to maintain
                privacy standards, a key differentiator for the company.
                """,
                "metadata": {
                    "ticker": ticker,
                    "document_type": "news",
                    "date": "2024-10-10",
                    "sentiment": "positive",
                    "source": "financial_news"
                }
            }
        ]

        # Add documents to RAG
        count = self.add_documents(sample_docs, "financial_reports")
        print(f"✅ Loaded {count} document chunks for {ticker}")

        return count

    def retrieve(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[Document]:
        """Retrieve relevant documents

        Args:
            query: Search query
            k: Number of documents to retrieve
            filter: Optional metadata filter

        Returns:
            List of relevant documents
        """
        if filter:
            # Some vector stores support metadata filtering
            try:
                return self.vector_store.similarity_search(
                    query, k=k, filter=filter
                )
            except:
                # Fallback to no filter
                return self.vector_store.similarity_search(query, k=k)
        else:
            return self.vector_store.similarity_search(query, k=k)

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Retrieve documents with relevance scores"""
        return self.vector_store.similarity_search_with_relevance_scores(query, k=k)

    def generate_with_context(self, query: str, context_docs: Optional[List[Document]] = None) -> str:
        """Generate response using retrieved context

        Args:
            query: User query
            context_docs: Optional pre-retrieved documents

        Returns:
            Generated response with citations
        """
        # Retrieve context if not provided
        if context_docs is None:
            context_docs = self.retrieve(query)

        # Build context string
        context_parts = []
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get("source", "unknown")
            doc_type = doc.metadata.get("document_type", "general")
            context_parts.append(f"[Source {i+1}: {doc_type} from {source}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # Create prompt
        prompt_template = """You are a financial analyst with access to relevant documents.

Context Documents:
{context}

Question: {question}

Instructions:
1. Answer based on the provided context
2. Cite sources using [Source N] format
3. If information is not in context, say so
4. Be specific and use numbers when available

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Generate response
        formatted_prompt = prompt.format(context=context, question=query)
        response = self.llm.invoke(formatted_prompt)

        return response

    def query_with_chain(self, query: str) -> Dict[str, Any]:
        """Use custom retrieval and generation

        This demonstrates a manual RAG pipeline without pre-built chains
        """
        # Retrieve documents
        docs = self.vector_store.similarity_search(query, k=5)

        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Create prompt
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        # Generate response
        result = self.llm.invoke(prompt)

        return {
            "answer": result,
            "source_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }

    def hybrid_search(self, query: str, k: int = 5) -> List[Document]:
        """Demonstrate hybrid search (semantic + keyword)

        This is a simplified demonstration of hybrid search
        """
        # Semantic search
        semantic_results = self.retrieve(query, k=k)

        # Keyword search (simplified - using basic matching)
        keyword_results = []
        all_docs = []

        # Get all documents (in practice, use a proper keyword index)
        for source_docs in self.document_sources.values():
            all_docs.extend(source_docs)

        # Simple keyword matching
        query_words = set(query.lower().split())
        for doc in all_docs:
            if hasattr(doc, 'page_content'):
                doc_words = set(doc.page_content.lower().split())
                overlap = len(query_words.intersection(doc_words))
                if overlap > 0:
                    keyword_results.append((doc, overlap))

        # Sort by overlap
        keyword_results.sort(key=lambda x: x[1], reverse=True)
        keyword_docs = [doc for doc, _ in keyword_results[:k]]

        # Combine results (remove duplicates)
        seen_contents = set()
        combined = []

        for doc in semantic_results + keyword_docs:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            if content not in seen_contents:
                seen_contents.add(content)
                combined.append(doc)

        return combined[:k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        stats = {
            "vector_store_type": self.vector_store_type,
            "total_documents": sum(len(docs) for docs in self.document_sources.values()),
            "document_sources": {k: len(v) for k, v in self.document_sources.items()},
            "embedding_model": self.embeddings.model,
            "llm_model": self.llm.model,
            "chunk_size": self.text_splitter._chunk_size,
            "chunk_overlap": self.text_splitter._chunk_overlap
        }

        return stats

# Demonstration function
def demonstrate_rag():
    """Demonstrate RAG capabilities"""
    print("🎓 RAG Engine Demonstration")
    print("=" * 60)

    # Create RAG engine
    rag = RAGEngine()

    # Load sample financial documents
    print("\n📚 Loading financial documents...")
    rag.load_financial_documents("AAPL")

    # Demonstrate different retrieval methods
    queries = [
        "What was Apple's revenue in Q3 2024?",
        "What are the main risks for Apple stock?",
        "What is the analyst price target?",
        "Tell me about AI initiatives"
    ]

    for query in queries:
        print(f"\n❓ Query: {query}")
        print("-" * 60)

        # Method 1: Retrieve and show documents
        print("\n📄 Retrieved Documents:")
        docs = rag.retrieve(query, k=3)
        for i, doc in enumerate(docs, 1):
            print(f"\n  Document {i}:")
            print(f"    Type: {doc.metadata.get('document_type', 'unknown')}")
            print(f"    Preview: {doc.page_content[:100]}...")

        # Method 2: Generate with context
        print("\n🤖 Generated Answer:")
        answer = rag.generate_with_context(query)
        print(f"  {answer}")

    # Show statistics
    print("\n📊 RAG System Statistics:")
    stats = rag.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    demonstrate_rag()
