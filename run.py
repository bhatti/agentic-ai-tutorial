#!/usr/bin/env python3
"""
Launcher script for Agentic AI Tutorial
This handles all path and import issues automatically.

Usage:
    python run.py
"""

import sys
import os
import subprocess

def main():
    """Run the Streamlit application with proper paths"""

    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    # Set PYTHONPATH environment variable
    env = os.environ.copy()
    env['PYTHONPATH'] = current_dir + os.pathsep + env.get('PYTHONPATH', '')

    print("🚀 Starting Agentic AI Tutorial...")
    print(f"📁 Working directory: {current_dir}")
    print("-" * 50)

    # Run streamlit with proper environment
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "src/main.py"],
            env=env,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
