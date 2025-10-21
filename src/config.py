"""Configuration for Ollama-based Financial Agent
This demonstrates local LLM usage with open-source models
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum

class OllamaModel(Enum):
    """Available open-source models in Ollama"""
    LLAMA3_8B = "llama3.2:latest"
    LLAMA3_70B = "llama3.2:70b"
    QWEN2_5 = "qwen2.5:latest"
    MISTRAL = "mistral:latest"
    MIXTRAL = "mixtral:latest"
    PHI3 = "phi3:latest"
    GEMMA2 = "gemma2:latest"
    DEEPSEEK = "deepseek-coder:latest"
    SOLAR = "solar:latest"

    # For embeddings
    NOMIC_EMBED = "nomic-embed-text:latest"
    MXBAI_EMBED = "mxbai-embed-large:latest"

@dataclass
class OllamaConfig:
    """Configuration for Ollama-based agents

    This config demonstrates:
    - Multiple model support for different tasks
    - Local embedding models for RAG
    - Temperature and parameter tuning
    - Tool configuration for MCP-style interactions
    """

    # Ollama connection
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Set environment variable for langchain-ollama
    os.environ["OLLAMA_BASE_URL"] = base_url

    # Model selection - demonstrating task-specific models
    reasoning_model: str = OllamaModel.LLAMA3_8B.value  # For ReAct reasoning
    analysis_model: str = OllamaModel.QWEN2_5.value    # For financial analysis
    coding_model: str = OllamaModel.DEEPSEEK.value     # For code generation
    embedding_model: str = OllamaModel.NOMIC_EMBED.value  # For RAG embeddings

    # Model parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    num_predict: int = 2048

    # RAG configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 5

    # Vector store settings
    vector_store_type: Literal["chroma", "faiss", "qdrant"] = "chroma"
    persist_directory: str = "./data/vector_store"

    # ReAct configuration
    max_reasoning_steps: int = 10
    enable_reasoning_trace: bool = True

    # Tool configuration (MCP-style)
    available_tools: List[str] = field(default_factory=lambda: [
        "web_search",
        "calculator",
        "code_interpreter",
        "file_reader",
        "sql_query",
        "api_caller"
    ])

    # Memory configuration
    memory_type: Literal["buffer", "summary", "kg"] = "buffer"
    max_memory_tokens: int = 2000

    # Logging and debugging
    verbose: bool = True
    log_level: str = "INFO"
    save_traces: bool = True
    trace_dir: str = "./data/traces"

    def get_model_config(self, task: str = "default") -> Dict[str, Any]:
        """Get model configuration for specific task"""

        model_map = {
            "reasoning": self.reasoning_model,
            "analysis": self.analysis_model,
            "coding": self.coding_model,
            "default": self.reasoning_model
        }

        return {
            "model": model_map.get(task, self.reasoning_model),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_predict": self.num_predict,
            "options": {
                "num_ctx": 4096,  # Context window
                "num_batch": 512,  # Batch size for prompt eval
                "num_gpu": 1,      # Number of GPUs to use
            }
        }

    def validate(self) -> bool:
        """Validate Ollama connection and models"""
        import requests

        try:
            # Check Ollama is running
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError(f"Ollama not responding at {self.base_url}")

            # Check required models are available
            available_models = [m["name"] for m in response.json().get("models", [])]
            required_models = [
                self.reasoning_model,
                self.analysis_model,
                self.embedding_model
            ]

            missing_models = [m for m in required_models if m not in available_models]

            if missing_models:
                print(f"⚠️ Missing models: {missing_models}")
                print("Pull them with: ollama pull <model_name>")
                return False

            return True

        except Exception as e:
            print(f"❌ Ollama validation failed: {e}")
            return False

# Global config instance
config = OllamaConfig()

# Educational helpers
def list_available_models():
    """List all models available in Ollama"""
    import requests

    try:
        response = requests.get(f"{config.base_url}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [m["name"] for m in models]
    except:
        return []

    return []

def pull_required_models():
    """Pull required models for the tutorial"""
    import subprocess

    required = [
        "llama3.2:latest",
        "qwen2.5:latest",
        "nomic-embed-text:latest"
    ]

    for model in required:
        print(f"Pulling {model}...")
        subprocess.run(["ollama", "pull", model])
