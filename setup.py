#!/usr/bin/env python3
"""Setup script for Agentic AI Tutorial with Ollama
This script helps users set up their environment correctly
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is suitable"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True

def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "--version"],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            return True
    except FileNotFoundError:
        pass

    print("❌ Ollama not found")
    print("\nTo install Ollama:")

    system = platform.system()
    if system == "Darwin":  # macOS
        print("  brew install ollama")
        print("  or download from: https://ollama.ai/download")
    elif system == "Linux":
        print("  curl -fsSL https://ollama.ai/install.sh | sh")
    elif system == "Windows":
        print("  Download from: https://ollama.ai/download")

    return False

def check_ollama_running():
    """Check if Ollama service is running"""
    import requests

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            return True
    except:
        pass

    print("⚠️ Ollama service not running")
    print("  Start it with: ollama serve")
    return False

def pull_required_models():
    """Pull required Ollama models"""
    models = [
        ("llama3.2", "Main reasoning model"),
        ("qwen2.5", "Alternative model"),
        ("nomic-embed-text", "Embedding model for RAG")
    ]

    print("\n📦 Pulling required models...")

    for model, description in models:
        print(f"\nPulling {model} ({description})...")
        try:
            # Check if model exists
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )

            if model in result.stdout:
                print(f"  ✅ {model} already available")
            else:
                # Pull model
                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=False,
                    text=True
                )
                if result.returncode == 0:
                    print(f"  ✅ {model} pulled successfully")
                else:
                    print(f"  ❌ Failed to pull {model}")
                    return False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    return True

def create_directories():
    """Create necessary directories"""
    dirs = [
        "data",
        "data/vector_store",
        "data/traces",
        "logs",
        "docs"
    ]

    print("\n📁 Creating directories...")
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")

def install_dependencies():
    """Install Python dependencies"""
    print("\n📚 Installing Python dependencies...")

    # Upgrade pip first
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install requirements
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        capture_output=False
    )

    if result.returncode == 0:
        print("✅ Dependencies installed")
        return True
    else:
        print("❌ Failed to install dependencies")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")

    if env_file.exists():
        print("✅ .env file exists")
        return

    print("\n📝 Creating .env file...")

    env_content = """# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Model Selection
LLM_PROVIDER=ollama
REASONING_MODEL=llama3.2:latest
ANALYSIS_MODEL=qwen2.5:latest
EMBEDDING_MODEL=nomic-embed-text:latest

# Optional: API Keys for external services
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here

# Application Settings
LOG_LEVEL=INFO
VECTOR_STORE_TYPE=chroma
"""

    env_file.write_text(env_content)
    print("✅ .env file created")

def test_setup():
    """Test the setup with a simple script"""
    print("\n🧪 Testing setup...")

    test_code = """
import sys
sys.path.insert(0, '.')

from src.config import OllamaConfig

config = OllamaConfig()
if config.validate():
    print("✅ Setup test passed!")
    print(f"   Using models: {config.reasoning_model}")
else:
    print("❌ Setup test failed")
"""

    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

    return result.returncode == 0

def main():
    """Main setup process"""
    print("🚀 Agentic AI Tutorial Setup")
    print("=" * 50)

    # Check prerequisites
    checks = [
        ("Python Version", check_python_version),
        ("Ollama Installation", check_ollama_installed),
    ]

    for name, check_func in checks:
        if not check_func():
            print(f"\n❌ Setup failed at: {name}")
            print("Please fix the issue and run setup again.")
            sys.exit(1)

    # Check if Ollama is running (warning only)
    ollama_running = check_ollama_running()

    # Create directories
    create_directories()

    # Create .env file
    create_env_file()

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies")
        sys.exit(1)

    # Pull models if Ollama is running
    if ollama_running:
        if not pull_required_models():
            print("\n⚠️ Some models failed to pull")
            print("You can pull them manually with: ollama pull <model>")
    else:
        print("\n⚠️ Skipping model pull (Ollama not running)")
        print("After starting Ollama, pull models with:")
        print("  ollama pull llama3.2")
        print("  ollama pull qwen2.5")
        print("  ollama pull nomic-embed-text")

    # Test setup
    if ollama_running:
        test_success = test_setup()
    else:
        test_success = False
        print("\n⚠️ Skipping test (Ollama not running)")

    # Final summary
    print("\n" + "=" * 50)
    print("📊 Setup Summary")
    print("=" * 50)

    if test_success:
        print("✅ Setup completed successfully!")
        print("\nYou can now run:")
        print("  streamlit run src/main.py")
        print("\nOr use the CLI:")
        print("  python src/cli.py --help")
    else:
        print("⚠️ Setup completed with warnings")
        print("\nBefore running the application:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull models (if needed)")
        print("3. Run: streamlit run src/main.py")

if __name__ == "__main__":
    main()
