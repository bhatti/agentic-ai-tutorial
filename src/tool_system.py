"""Tool System demonstrating MCP (Model Context Protocol) patterns
This shows how to create a flexible tool system for AI agents with:
- Tool registration and discovery
- Input/output schemas
- Error handling
- Context management
"""

from typing import Dict, Any, Callable, Optional, List, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import inspect
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

class ToolCategory(Enum):
    """Categories of tools for organization"""
    DATA_RETRIEVAL = "data_retrieval"
    CALCULATION = "calculation"
    WEB_INTERACTION = "web_interaction"
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    API_INTEGRATION = "api_integration"
    ANALYSIS = "analysis"

class ToolParameter(BaseModel):
    """Defines a tool parameter with validation"""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None

class ToolSchema(BaseModel):
    """Schema definition for a tool (MCP-style)"""
    name: str
    description: str
    category: ToolCategory
    parameters: List[ToolParameter]
    returns: str  # Description of return value
    examples: List[Dict[str, Any]] = Field(default_factory=list)

class BaseTool(ABC):
    """Abstract base class for tools"""

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Return tool schema"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        pass

    def validate_input(self, **kwargs) -> bool:
        """Validate input against schema"""
        schema = self.get_schema()

        for param in schema.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"Missing required parameter: {param.name}")

            if param.enum and kwargs.get(param.name) not in param.enum:
                raise ValueError(f"Invalid value for {param.name}. Must be one of {param.enum}")

        return True

# Example Tool Implementations

class StockDataTool(BaseTool):
    """Tool for fetching stock market data"""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="stock_data",
            description="Fetch real-time stock market data for a given ticker",
            category=ToolCategory.DATA_RETRIEVAL,
            parameters=[
                ToolParameter(
                    name="ticker",
                    type="string",
                    description="Stock ticker symbol (e.g., AAPL, GOOGL)",
                    required=True
                ),
                ToolParameter(
                    name="metrics",
                    type="array",
                    description="Specific metrics to retrieve",
                    required=False,
                    default=["price", "volume", "market_cap"]
                )
            ],
            returns="Dictionary containing requested stock metrics",
            examples=[
                {"ticker": "AAPL", "metrics": ["price", "pe_ratio"]},
                {"ticker": "TSLA"}
            ]
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Fetch stock data using yfinance"""
        import yfinance as yf

        self.validate_input(**kwargs)

        ticker = kwargs["ticker"].upper()
        metrics = kwargs.get("metrics", ["price", "volume", "market_cap"])

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            result = {"ticker": ticker}

            metric_mapping = {
                "price": "currentPrice",
                "volume": "volume",
                "market_cap": "marketCap",
                "pe_ratio": "trailingPE",
                "dividend_yield": "dividendYield",
                "52_week_high": "fiftyTwoWeekHigh",
                "52_week_low": "fiftyTwoWeekLow"
            }

            for metric in metrics:
                yf_key = metric_mapping.get(metric, metric)
                result[metric] = info.get(yf_key, None)

            return result

        except Exception as e:
            return {"error": str(e), "ticker": ticker}

class CalculatorTool(BaseTool):
    """Tool for mathematical calculations"""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="calculator",
            description="Perform mathematical calculations",
            category=ToolCategory.CALCULATION,
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Mathematical expression to evaluate",
                    required=True
                ),
                ToolParameter(
                    name="precision",
                    type="number",
                    description="Number of decimal places",
                    required=False,
                    default=2
                )
            ],
            returns="Calculated result as a number",
            examples=[
                {"expression": "2 + 2"},
                {"expression": "(150 - 120) / 120 * 100", "precision": 1}
            ]
        )

    def execute(self, **kwargs) -> float:
        """Safely evaluate mathematical expression"""
        self.validate_input(**kwargs)

        expression = kwargs["expression"]
        precision = kwargs.get("precision", 2)

        # Safe evaluation with limited scope
        try:
            # Remove potentially dangerous operations
            safe_expr = expression.replace("__", "").replace("import", "")

            # Define allowed functions
            safe_dict = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "pow": pow, "len": len
            }

            # Add math functions
            import math
            for name in ["sqrt", "log", "log10", "sin", "cos", "tan", "pi", "e"]:
                if hasattr(math, name):
                    safe_dict[name] = getattr(math, name)

            result = eval(safe_expr, {"__builtins__": {}}, safe_dict)

            return round(result, precision)

        except Exception as e:
            raise ValueError(f"Calculation error: {e}")

class WebSearchTool(BaseTool):
    """Tool for web search (mock implementation)"""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="web_search",
            description="Search the web for information",
            category=ToolCategory.WEB_INTERACTION,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True
                ),
                ToolParameter(
                    name="num_results",
                    type="number",
                    description="Number of results to return",
                    required=False,
                    default=5
                )
            ],
            returns="List of search results with title, snippet, and URL",
            examples=[
                {"query": "Apple earnings Q3 2024"},
                {"query": "Tesla stock analysis", "num_results": 10}
            ]
        )

    def execute(self, **kwargs) -> List[Dict[str, str]]:
        """Mock web search implementation"""
        self.validate_input(**kwargs)

        query = kwargs["query"]
        num_results = kwargs.get("num_results", 5)

        # Mock results for demonstration
        mock_results = [
            {
                "title": f"Result {i+1} for: {query}",
                "snippet": f"This is a snippet about {query}. It contains relevant information...",
                "url": f"https://example.com/result{i+1}"
            }
            for i in range(min(num_results, 5))
        ]

        return mock_results

class SQLQueryTool(BaseTool):
    """Tool for SQL database queries"""

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="sql_query",
            description="Execute SQL queries on the database",
            category=ToolCategory.DATABASE,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="SQL query to execute",
                    required=True
                ),
                ToolParameter(
                    name="database",
                    type="string",
                    description="Database name",
                    required=False,
                    default="financial_data"
                )
            ],
            returns="Query results as list of dictionaries",
            examples=[
                {"query": "SELECT * FROM stocks WHERE pe_ratio < 20"},
                {"query": "SELECT ticker, price FROM stocks ORDER BY volume DESC LIMIT 10"}
            ]
        )

    def execute(self, **kwargs) -> List[Dict[str, Any]]:
        """Execute SQL query (mock implementation)"""
        self.validate_input(**kwargs)

        # In real implementation, would connect to actual database
        # This is a mock for demonstration

        query = kwargs["query"].lower()

        if "stocks" in query:
            return [
                {"ticker": "AAPL", "price": 175.50, "pe_ratio": 28.5},
                {"ticker": "GOOGL", "price": 140.25, "pe_ratio": 25.2},
                {"ticker": "MSFT", "price": 380.75, "pe_ratio": 32.1}
            ]

        return []

class ToolRegistry:
    """Registry for managing tools (MCP-style tool management)"""

    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.categories: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }

    def register(self, tool: BaseTool):
        """Register a tool"""
        schema = tool.get_schema()
        self.tools[schema.name] = tool
        self.categories[schema.category].append(schema.name)
        return self

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(name)

    def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        return tool.execute(**kwargs)

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[str]:
        """List available tools"""
        if category:
            return self.categories.get(category, [])
        return list(self.tools.keys())

    def get_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get all tool schemas"""
        return {
            name: tool.get_schema().dict()
            for name, tool in self.tools.items()
        }

    def get_tool_description_for_llm(self) -> str:
        """Get formatted tool descriptions for LLM prompts"""
        descriptions = []

        for name, tool in self.tools.items():
            schema = tool.get_schema()

            params_str = ", ".join([
                f"{p.name}: {p.type}" + (" (optional)" if not p.required else "")
                for p in schema.parameters
            ])

            descriptions.append(
                f"{name}({params_str}): {schema.description}"
            )

        return "\n".join(descriptions)

class ToolExecutor:
    """Executes tools with enhanced context management"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.execution_history: List[Dict[str, Any]] = []

    def execute_with_context(self,
                            tool_name: str,
                            parameters: Dict[str, Any],
                            context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute tool with context tracking"""

        # Record execution attempt
        execution_record = {
            "tool": tool_name,
            "parameters": parameters,
            "context": context,
            "timestamp": json.dumps(None),  # Would use datetime in real impl
            "status": "pending"
        }

        try:
            # Execute tool
            result = self.registry.execute(tool_name, **parameters)

            # Update record
            execution_record["status"] = "success"
            execution_record["result"] = result

        except Exception as e:
            # Handle error
            execution_record["status"] = "error"
            execution_record["error"] = str(e)
            result = {"error": str(e)}

        # Save to history
        self.execution_history.append(execution_record)

        return result

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get tool execution history"""
        return self.execution_history

    def create_tool_chain(self, steps: List[Tuple[str, Dict[str, Any]]]) -> List[Any]:
        """Execute a chain of tools where each output feeds the next"""
        results = []
        previous_result = None

        for tool_name, params in steps:
            # Allow referencing previous result
            if previous_result is not None:
                params = self._inject_previous_result(params, previous_result)

            result = self.execute_with_context(tool_name, params)
            results.append(result)
            previous_result = result

        return results

    def _inject_previous_result(self, params: Dict[str, Any], previous_result: Any) -> Dict[str, Any]:
        """Inject previous result into parameters"""
        # Simple implementation - replace $PREVIOUS with previous result
        params_str = json.dumps(params)
        if "$PREVIOUS" in params_str:
            params_str = params_str.replace("$PREVIOUS", json.dumps(previous_result))
            return json.loads(params_str)
        return params

# Demonstration function
def demonstrate_tools():
    """Demonstrate the tool system"""
    print("🎓 Tool System Demonstration (MCP-style)")
    print("=" * 60)

    # Create registry
    registry = ToolRegistry()

    # Register tools
    registry.register(StockDataTool())
    registry.register(CalculatorTool())
    registry.register(WebSearchTool())
    registry.register(SQLQueryTool())

    # Show available tools
    print("\n📦 Available Tools:")
    for category in ToolCategory:
        tools = registry.list_tools(category)
        if tools:
            print(f"\n  {category.value}:")
            for tool in tools:
                print(f"    - {tool}")

    # Show tool descriptions for LLM
    print("\n📝 Tool Descriptions for LLM:")
    print(registry.get_tool_description_for_llm())

    # Execute some tools
    executor = ToolExecutor(registry)

    print("\n🔧 Tool Execution Examples:")

    # Example 1: Stock data
    print("\n1. Fetching stock data:")
    result = executor.execute_with_context(
        "stock_data",
        {"ticker": "AAPL", "metrics": ["price", "pe_ratio", "market_cap"]}
    )
    print(f"   Result: {result}")

    # Example 2: Calculator
    print("\n2. Calculating potential return:")
    result = executor.execute_with_context(
        "calculator",
        {"expression": "(200 - 175) / 175 * 100", "precision": 2}
    )
    print(f"   Result: {result}% potential return")

    # Example 3: Tool chain
    print("\n3. Tool chain example:")
    chain_results = executor.create_tool_chain([
        ("stock_data", {"ticker": "AAPL"}),
        ("web_search", {"query": "AAPL latest news", "num_results": 3})
    ])
    print(f"   Chain completed with {len(chain_results)} results")

    # Show execution history
    print("\n📊 Execution History:")
    for record in executor.get_execution_history()[-3:]:
        print(f"   Tool: {record['tool']}, Status: {record['status']}")

if __name__ == "__main__":
    demonstrate_tools()
