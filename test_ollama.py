#!/usr/bin/env python3
"""
Test Ollama connection and models
Run this to verify Ollama is working properly
"""

import os
import sys

def test_ollama_direct():
    """Test Ollama directly"""
    print("Testing Ollama connection...")

    try:
        import ollama

        # Test connection
        print("✅ Ollama package imported")

        # List models
        models = ollama.list()
        print(f"✅ Found {len(models['models'])} models:")
        for model in models['models']:
            print(f"   - {model['name']}")

        # Test generation
        response = ollama.generate(model='llama3.2:latest', prompt='Say hello')
        print(f"✅ Generation test: {response['response'][:50]}...")

        return True

    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

def test_langchain_ollama():
    """Test LangChain Ollama integration"""
    print("\nTesting LangChain Ollama integration...")

    try:
        from langchain_ollama import OllamaLLM

        # Set environment variable
        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

        # Create LLM
        llm = OllamaLLM(model="llama3.2:latest")
        print("✅ LangChain Ollama LLM created")

        # Test generation
        response = llm.invoke("What is 2+2?")
        print(f"✅ LangChain generation: {response[:100]}...")

        return True

    except Exception as e:
        print(f"❌ LangChain Ollama test failed: {e}")
        return False

def test_embeddings():
    """Test Ollama embeddings"""
    print("\nTesting Ollama embeddings...")

    try:
        from langchain_ollama import OllamaEmbeddings

        # Create embeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        print("✅ Embeddings model created")

        # Test embedding
        result = embeddings.embed_query("test text")
        print(f"✅ Embedding generated: {len(result)} dimensions")

        return True

    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        print("   Try: ollama pull nomic-embed-text")
        return False

def main():
    print("=" * 60)
    print("🧪 Ollama Connection Test")
    print("=" * 60)

    # Check if Ollama is running
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("❌ Ollama is not responding!")
            print("   Run: ollama serve")
            sys.exit(1)
    except:
        print("❌ Cannot connect to Ollama!")
        print("   Make sure Ollama is running: ollama serve")
        sys.exit(1)

    tests = [
        test_ollama_direct,
        test_langchain_ollama,
        test_embeddings
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    if all(results):
        print("✅ All tests passed! Ollama is working correctly.")
        print("\nYou can run the app with: python run.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull models: ollama pull llama3.2")
        print("3. Pull embeddings: ollama pull nomic-embed-text")

if __name__ == "__main__":
    main()
