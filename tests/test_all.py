#!/usr/bin/env python3
"""
Simple test file to verify all components work
Run with: python test_all.py
"""

import sys
import os
sys.path.insert(0, '.')

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        from src.config import OllamaConfig
        print("✅ Config module")

        from src.react_agent import ReActAgent
        print("✅ ReAct module")

        from src.rag_engine import RAGEngine
        print("✅ RAG module")

        from src.tool_system import ToolRegistry
        print("✅ Tool system")

        from src.workflow import FinancialAgentWorkflow
        print("✅ Workflow module")

        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_ollama_connection():
    """Test Ollama connection"""
    print("\nTesting Ollama connection...")

    try:
        from src.config import OllamaConfig
        config = OllamaConfig()

        if config.validate():
            print("✅ Ollama is running and models are available")
            return True
        else:
            print("❌ Ollama validation failed")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_react_basic():
    """Test basic ReAct functionality"""
    print("\nTesting ReAct agent...")

    try:
        from src.react_agent import ReActAgent

        agent = ReActAgent(verbose=False)

        # Simple calculation test
        answer, trace = agent.think("Calculate 15% of 200")

        if trace and answer:
            print(f"✅ ReAct working - Answer: {answer}")
            return True
        else:
            print("❌ ReAct failed to produce answer")
            return False

    except Exception as e:
        print(f"❌ ReAct test failed: {e}")
        return False

def test_rag_basic():
    """Test basic RAG functionality"""
    print("\nTesting RAG engine...")

    try:
        from src.rag_engine import RAGEngine

        rag = RAGEngine()

        # Add a simple document
        docs = [{
            "content": "Apple Inc. reported revenue of $94.9 billion in Q3 2024.",
            "metadata": {"source": "test"}
        }]

        count = rag.add_documents(docs)

        if count > 0:
            print(f"✅ RAG working - Added {count} chunks")
            return True
        else:
            print("❌ RAG failed to add documents")
            return False

    except Exception as e:
        print(f"❌ RAG test failed: {e}")
        return False

def test_tools_basic():
    """Test basic tool functionality"""
    print("\nTesting tool system...")

    try:
        from src.tool_system import ToolRegistry, CalculatorTool

        registry = ToolRegistry()
        registry.register(CalculatorTool())

        result = registry.execute("calculator", expression="2+2", precision=0)

        if result == 4:
            print(f"✅ Tools working - Calculator: 2+2={result}")
            return True
        else:
            print(f"❌ Tool calculation wrong: {result}")
            return False

    except Exception as e:
        print(f"❌ Tool test failed: {e}")
        return False

def test_workflow_basic():
    """Test basic workflow initialization"""
    print("\nTesting workflow...")

    try:
        from src.workflow import FinancialAgentWorkflow, AgentState

        workflow = FinancialAgentWorkflow()

        # Check that workflow compiled
        if workflow.app:
            print("✅ Workflow compiled successfully")
            return True
        else:
            print("❌ Workflow compilation failed")
            return False

    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("🧪 Agentic AI Tutorial - System Test")
    print("=" * 50)

    tests = [
        ("Imports", test_imports),
        ("Ollama Connection", test_ollama_connection),
        ("ReAct Agent", test_react_basic),
        ("RAG Engine", test_rag_basic),
        ("Tool System", test_tools_basic),
        ("Workflow", test_workflow_basic)
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name:20} {status}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed == total:
        print("\n🎉 All tests passed! System ready.")
        print("\nYou can now run:")
        print("  streamlit run src/main.py")
        print("  python src/cli.py --help")
    else:
        print("\n⚠️ Some tests failed. Please check:")
        print("1. Is Ollama running? (ollama serve)")
        print("2. Are models pulled? (ollama pull llama3.2)")
        print("3. Are all dependencies installed? (pip install -r requirements.txt)")

    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
