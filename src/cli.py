#!/usr/bin/env python3
"""Command-line interface for Agentic AI Tutorial
Allows running demonstrations without the Streamlit UI
"""

import argparse
import sys
import json
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax

# Add src to path
sys.path.insert(0, '.')

from src.config import OllamaConfig
from src.react_agent import ReActAgent
from src.rag_engine import RAGEngine
from src.tool_system import ToolRegistry, ToolExecutor, StockDataTool, CalculatorTool
from src.workflow import FinancialAgentWorkflow

console = Console()

class AgenticCLI:
    """CLI for demonstrating agentic AI patterns"""

    def __init__(self):
        self.config = OllamaConfig()

    def demo_react(self, question: str, verbose: bool = False):
        """Demonstrate ReAct reasoning"""
        console.print(Panel("[bold cyan]ReAct Demo: Reasoning and Acting[/bold cyan]"))

        with console.status("[bold green]Agent thinking...") as status:
            agent = ReActAgent(verbose=verbose)
            answer, trace = agent.think(question)

        # Display result
        console.print(f"\n[bold]Question:[/bold] {question}")
        console.print(f"[bold]Answer:[/bold] {answer}\n")

        # Show reasoning trace
        if verbose or len(trace) > 0:
            table = Table(title="Reasoning Trace")
            table.add_column("Step", style="cyan", no_wrap=True)
            table.add_column("Thought", style="yellow")
            table.add_column("Action", style="green")
            table.add_column("Observation", style="blue")

            for i, step in enumerate(trace, 1):
                obs_str = str(step.observation)[:50] + "..." if len(str(step.observation)) > 50 else str(step.observation)
                table.add_row(
                    str(i),
                    step.thought[:50] + "..." if len(step.thought) > 50 else step.thought,
                    f"{step.action}({step.action_input})",
                    obs_str
                )

            console.print(table)

    def demo_rag(self, ticker: str, query: str):
        """Demonstrate RAG"""
        console.print(Panel("[bold cyan]RAG Demo: Retrieval Augmented Generation[/bold cyan]"))

        rag = RAGEngine()

        # Load documents
        with console.status(f"[bold green]Loading documents for {ticker}..."):
            count = rag.load_financial_documents(ticker)

        console.print(f"✅ Loaded {count} document chunks\n")

        # Retrieve and generate
        with console.status("[bold green]Retrieving and generating..."):
            docs = rag.retrieve_with_scores(query, k=3)
            answer = rag.generate_with_context(query)

        # Display results
        console.print(f"[bold]Query:[/bold] {query}\n")
        console.print(Panel(answer, title="Generated Answer", border_style="green"))

        # Show retrieved documents
        console.print("\n[bold]Retrieved Documents:[/bold]")
        for i, (doc, score) in enumerate(docs, 1):
            console.print(f"\n{i}. [cyan]Score: {score:.3f}[/cyan]")
            console.print(f"   Type: {doc.metadata.get('document_type', 'unknown')}")
            console.print(f"   Preview: {doc.page_content[:100]}...")

    def demo_tools(self, tool_name: str, params: Dict[str, Any]):
        """Demonstrate tool usage"""
        console.print(Panel("[bold cyan]Tools Demo: MCP-style Tool System[/bold cyan]"))

        registry = ToolRegistry()
        registry.register(StockDataTool())
        registry.register(CalculatorTool())

        executor = ToolExecutor(registry)

        # Execute tool
        with console.status(f"[bold green]Executing {tool_name}..."):
            result = executor.execute_with_context(tool_name, params)

        # Display result
        console.print(f"\n[bold]Tool:[/bold] {tool_name}")
        console.print(f"[bold]Parameters:[/bold]")
        console.print(json.dumps(params, indent=2))
        console.print(f"\n[bold]Result:[/bold]")

        if isinstance(result, dict) or isinstance(result, list):
            console.print(Syntax(json.dumps(result, indent=2), "json"))
        else:
            console.print(result)

    def demo_workflow(self, ticker: str, query: Optional[str] = None):
        """Demonstrate complete workflow"""
        console.print(Panel("[bold cyan]Workflow Demo: Complete Agent Analysis[/bold cyan]"))

        workflow = FinancialAgentWorkflow()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Analyzing {ticker}...", total=None)

            result = workflow.analyze(ticker, query)

        # Display results
        console.print(f"\n[bold]Analysis for {ticker}[/bold]\n")

        # Key metrics
        table = Table(title="Key Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Recommendation", result.get("recommendation", "N/A"))
        table.add_row("Confidence", f"{result.get('confidence_score', 0):.1%}")

        market_data = result.get("market_data", {})
        table.add_row("Current Price", f"${market_data.get('price', 0):.2f}")

        console.print(table)

        # Report preview
        report = result.get("report", "No report generated")
        console.print(Panel(report[:500] + "...", title="Report Preview", border_style="blue"))

        # Summary stats
        console.print(f"\n[bold]Analysis Summary:[/bold]")
        console.print(f"  • Reasoning steps: {len(result.get('reasoning_trace', []))}")
        console.print(f"  • Documents retrieved: {len(result.get('retrieved_documents', []))}")
        console.print(f"  • Sentiment score: {result.get('sentiment_analysis', {}).get('overall', 'N/A')}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Agentic AI Tutorial CLI - Learn AI patterns with Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ReAct reasoning demo
  python src/cli.py react "What is Apple's P/E ratio and is it overvalued?"

  # RAG demo
  python src/cli.py rag AAPL "What are the key risks?"

  # Tool usage demo
  python src/cli.py tool stock_data --params '{"ticker": "AAPL"}'

  # Complete workflow
  python src/cli.py workflow AAPL --query "Should I invest?"

  # List available models
  python src/cli.py models
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ReAct command
    react_parser = subparsers.add_parser("react", help="Demo ReAct reasoning")
    react_parser.add_argument("question", help="Question to reason about")
    react_parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed trace")

    # RAG command
    rag_parser = subparsers.add_parser("rag", help="Demo RAG system")
    rag_parser.add_argument("ticker", help="Stock ticker for documents")
    rag_parser.add_argument("query", help="Query to answer")

    # Tool command
    tool_parser = subparsers.add_parser("tool", help="Demo tool usage")
    tool_parser.add_argument("name", help="Tool name",
                            choices=["stock_data", "calculator", "web_search"])
    tool_parser.add_argument("--params", type=json.loads, required=True,
                            help="Tool parameters as JSON")

    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Demo complete workflow")
    workflow_parser.add_argument("ticker", help="Stock ticker to analyze")
    workflow_parser.add_argument("--query", help="Specific query (optional)")

    # Models command
    models_parser = subparsers.add_parser("models", help="List available Ollama models")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check system setup")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize CLI
    cli = AgenticCLI()

    # Execute command
    try:
        if args.command == "react":
            cli.demo_react(args.question, args.verbose)

        elif args.command == "rag":
            cli.demo_rag(args.ticker.upper(), args.query)

        elif args.command == "tool":
            cli.demo_tools(args.name, args.params)

        elif args.command == "workflow":
            cli.demo_workflow(args.ticker.upper(), args.query)

        elif args.command == "models":
            import requests
            response = requests.get(f"{cli.config.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])

                table = Table(title="Available Ollama Models")
                table.add_column("Model", style="cyan")
                table.add_column("Size", style="green")
                table.add_column("Modified", style="yellow")

                for model in models:
                    size_gb = model.get("size", 0) / (1024**3)
                    table.add_row(
                        model["name"],
                        f"{size_gb:.1f} GB",
                        model.get("modified_at", "")[:10]
                    )

                console.print(table)
            else:
                console.print("[red]Failed to get models from Ollama[/red]")

        elif args.command == "check":
            console.print(Panel("[bold]System Check[/bold]"))

            # Check Ollama
            if cli.config.validate():
                console.print("✅ Ollama is running and models are available")
            else:
                console.print("❌ Ollama check failed")

            # Show configuration
            console.print(f"\nConfiguration:")
            console.print(f"  Base URL: {cli.config.base_url}")
            console.print(f"  Reasoning Model: {cli.config.reasoning_model}")
            console.print(f"  Analysis Model: {cli.config.analysis_model}")
            console.print(f"  Embedding Model: {cli.config.embedding_model}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
