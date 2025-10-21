"""ReAct (Reasoning and Acting) Agent Implementation
This demonstrates the ReAct pattern where the agent:
1. Thinks about what to do (Thought)
2. Takes an action (Action)
3. Observes the result (Observation)
4. Repeats until task is complete
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class ReActStep:
    """Represents a single step in ReAct reasoning"""

    def __init__(self, thought: str, action: str, action_input: str, observation: str = ""):
        self.thought = thought
        self.action = action
        self.action_input = action_input
        self.observation = observation

    def to_dict(self) -> Dict[str, str]:
        return {
            "thought": self.thought,
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation
        }

class ReActAgent:
    """ReAct Agent that demonstrates reasoning traces

    This implementation shows:
    - Explicit reasoning steps
    - Tool use with observations
    - Iterative problem solving
    - Transparent decision making
    """

    # ReAct prompt template
    REACT_PROMPT = """You are a financial analysis agent that uses the ReAct framework to solve problems.

You have access to the following tools:

{tools_description}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to ALWAYS follow the format exactly.

Question: {question}
Thought: {scratchpad}"""

    def __init__(self, llm: Optional[Ollama] = None, tools: Dict[str, Any] = None,
                 max_steps: int = 10, verbose: bool = True):
        """Initialize ReAct agent

        Args:
            llm: Ollama LLM instance
            tools: Dictionary of tool_name -> tool_function
            max_steps: Maximum reasoning steps
            verbose: Whether to print reasoning trace
        """
        try:
            from src.config import config
        except ModuleNotFoundError:
            from config import config

        self.llm = llm or Ollama(
            model=config.reasoning_model,
            temperature=0.3,  # Lower temperature for more focused reasoning
        )

        self.tools = tools or self._get_default_tools()
        self.max_steps = max_steps
        self.verbose = verbose
        self.reasoning_trace: List[ReActStep] = []

    def _get_default_tools(self) -> Dict[str, Any]:
        """Get default tools for demonstration"""
        return {
            "market_data": self._fetch_market_data,
            "calculate": self._calculate,
            "web_search": self._web_search,
            "analyze_sentiment": self._analyze_sentiment,
            "get_financial_ratios": self._get_financial_ratios,
            "final_answer": lambda x: x  # Special tool to indicate completion
        }

    def _get_tools_description(self) -> str:
        """Generate tool descriptions for the prompt"""
        descriptions = []

        tool_docs = {
            "market_data": "Fetch current market data for a stock ticker. Input: ticker symbol",
            "calculate": "Perform mathematical calculations. Input: mathematical expression",
            "web_search": "Search the web for information. Input: search query",
            "analyze_sentiment": "Analyze sentiment of text or news. Input: text to analyze",
            "get_financial_ratios": "Get financial ratios for a company. Input: ticker symbol",
            "final_answer": "Provide the final answer. Input: the final answer text"
        }

        for tool_name, doc in tool_docs.items():
            if tool_name in self.tools:
                descriptions.append(f"{tool_name}: {doc}")

        return "\n".join(descriptions)

    def think(self, question: str, context: Dict[str, Any] = None) -> Tuple[str, List[ReActStep]]:
        """Execute ReAct reasoning loop

        Args:
            question: The question to answer
            context: Optional context dictionary

        Returns:
            Tuple of (final_answer, reasoning_trace)
        """
        self.reasoning_trace = []
        scratchpad = ""

        if self.verbose:
            print(f"\n🤔 Starting ReAct reasoning for: {question}")

        for step in range(self.max_steps):
            # Build prompt with current scratchpad
            prompt = PromptTemplate(
                template=self.REACT_PROMPT,
                input_variables=["question", "tools_description", "tool_names", "scratchpad"]
            )

            formatted_prompt = prompt.format(
                question=question,
                tools_description=self._get_tools_description(),
                tool_names=", ".join(self.tools.keys()),
                scratchpad=scratchpad
            )

            if self.verbose:
                print(f"\n--- Step {step + 1} ---")

            # Get LLM response
            response = self.llm.invoke(formatted_prompt)

            if self.verbose:
                print(f"LLM Response: {response[:200]}...")

            # Parse response
            thought, action, action_input, is_final = self._parse_response(response)

            if is_final:
                # Found final answer
                if self.verbose:
                    print(f"\n✅ Final Answer: {action_input}")
                return action_input, self.reasoning_trace

            # Execute action
            observation = "No observation"
            if action in self.tools:
                try:
                    if self.verbose:
                        print(f"Executing: {action}({action_input})")
                    observation = self.tools[action](action_input)
                    if self.verbose:
                        print(f"Observation: {observation}")
                except Exception as e:
                    observation = f"Error executing {action}: {str(e)}"
                    if self.verbose:
                        print(f"Error: {observation}")
            else:
                observation = f"Unknown action: {action}. Available: {list(self.tools.keys())}"

            # Record step
            step_record = ReActStep(thought, action, action_input, str(observation))
            self.reasoning_trace.append(step_record)

            # Update scratchpad for next iteration
            scratchpad += f"{thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}\nThought: "

        # If we didn't get a final answer, provide a default
        if self.reasoning_trace:
            # Try to synthesize from observations
            observations = [step.observation for step in self.reasoning_trace]
            final_answer = f"Based on my analysis: {observations[-1] if observations else 'Unable to determine'}"
        else:
            final_answer = "Unable to determine answer within step limit"

        return final_answer, self.reasoning_trace

    def _parse_response(self, response: str) -> Tuple[str, str, str, bool]:
        """Parse LLM response to extract thought, action, and input

        Returns:
            Tuple of (thought, action, action_input, is_final_answer)
        """
        # Clean response
        response = response.strip()

        # Check for final answer
        if "Final Answer:" in response:
            parts = response.split("Final Answer:")
            thought = parts[0].strip()
            final_answer = parts[1].strip()
            return thought, "final_answer", final_answer, True

        # Parse thought, action, and action input
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL)
        action_match = re.search(r"Action:\s*(.+?)(?=Action Input:|$)", response, re.DOTALL)
        input_match = re.search(r"Action Input:\s*(.+?)(?=Observation:|$)", response, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else "Thinking..."
        action = action_match.group(1).strip() if action_match else "unknown"
        action_input = input_match.group(1).strip() if input_match else ""

        return thought, action, action_input, False

    # Tool implementations (simplified for demonstration)

    def _fetch_market_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch market data for a ticker"""
        import yfinance as yf

        try:
            stock = yf.Ticker(ticker.upper())
            info = stock.info

            # Check if we got valid data
            if not info or len(info) <= 1:
                # Try alternative method - get basic quote
                hist = stock.history(period="1d")
                if not hist.empty:
                    return {
                        "ticker": ticker.upper(),
                        "price": float(hist['Close'].iloc[-1]),
                        "volume": float(hist['Volume'].iloc[-1]),
                        "message": "Limited data available",
                        "data_source": "price_history"
                    }
                return {"error": f"No data available for {ticker}"}

            # Extract available data with fallbacks
            result = {"ticker": ticker.upper()}

            # Price data
            price_fields = ["currentPrice", "regularMarketPrice", "price"]
            for field in price_fields:
                if field in info and info[field] is not None:
                    result["price"] = info[field]
                    break

            # If still no price, try history
            if "price" not in result:
                hist = stock.history(period="1d")
                if not hist.empty:
                    result["price"] = float(hist['Close'].iloc[-1])

            # Other metrics with safe access
            result["market_cap"] = info.get("marketCap", "N/A")
            result["pe_ratio"] = info.get("trailingPE", "N/A")
            result["52_week_high"] = info.get("fiftyTwoWeekHigh", "N/A")
            result["52_week_low"] = info.get("fiftyTwoWeekLow", "N/A")
            result["volume"] = info.get("volume", info.get("regularMarketVolume", "N/A"))

            return result

        except Exception as e:
            # Try one more fallback - just get price from history
            try:
                stock = yf.Ticker(ticker.upper())
                hist = stock.history(period="5d")
                if not hist.empty:
                    return {
                        "ticker": ticker.upper(),
                        "price": float(hist['Close'].iloc[-1]),
                        "volume": float(hist['Volume'].iloc[-1]),
                        "error": f"Limited data: {str(e)}",
                        "data_source": "price_history_fallback"
                    }
            except:
                pass

            return {"error": str(e), "ticker": ticker}

    def _calculate(self, expression: str) -> float:
        """Safe mathematical calculation"""
        try:
            # Remove any dangerous operations
            safe_expr = expression.replace("__", "").replace("import", "")
            # Only allow basic math operations
            allowed_names = {
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len, "pow": pow
            }
            return eval(safe_expr, {"__builtins__": {}}, allowed_names)
        except Exception as e:
            return f"Calculation error: {e}"

    def _web_search(self, query: str) -> str:
        """Simulate web search (in real implementation, use actual search API)"""
        # This is a mock implementation for demonstration
        mock_results = {
            "earnings": "Company reported strong Q3 earnings with 15% revenue growth",
            "outlook": "Analysts maintain positive outlook with average price target of $150",
            "risks": "Key risks include regulatory challenges and market competition"
        }

        for key, result in mock_results.items():
            if key in query.lower():
                return result

        return "No specific information found for the query"

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text using local LLM"""
        prompt = f"Analyze the sentiment of this text and return a score from -1 (very negative) to 1 (very positive): {text}"

        response = self.llm.invoke(prompt)

        # Parse response (simplified)
        try:
            # Extract number from response
            import re
            numbers = re.findall(r'-?\d*\.?\d+', response)
            if numbers:
                score = float(numbers[0])
                score = max(-1, min(1, score))  # Clamp to [-1, 1]
            else:
                score = 0.0

            sentiment = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"

            return {
                "score": score,
                "sentiment": sentiment,
                "confidence": 0.85
            }
        except:
            return {"score": 0.0, "sentiment": "neutral", "confidence": 0.5}

    def _get_financial_ratios(self, ticker: str) -> Dict[str, float]:
        """Get financial ratios for a company"""
        import yfinance as yf

        try:
            stock = yf.Ticker(ticker.upper())
            info = stock.info

            return {
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "peg_ratio": info.get("pegRatio", 0),
                "price_to_book": info.get("priceToBook", 0),
                "debt_to_equity": info.get("debtToEquity", 0),
                "roe": info.get("returnOnEquity", 0),
                "roa": info.get("returnOnAssets", 0),
                "profit_margin": info.get("profitMargins", 0)
            }
        except Exception as e:
            return {"error": str(e)}

# Demonstration function
def demonstrate_react():
    """Demonstrate ReAct agent capabilities"""
    print("🎓 ReAct Agent Demonstration")
    print("=" * 60)

    # Create agent
    agent = ReActAgent(verbose=True)

    # Example questions
    questions = [
        "What is Apple's current P/E ratio and is it overvalued?",
        "Calculate the potential return if AAPL goes from current price to $200",
        "What are the main risks for Tesla stock?"
    ]

    for question in questions[:1]:  # Run first question as demo
        print(f"\n📊 Question: {question}")
        print("-" * 60)

        answer, trace = agent.think(question)

        print("\n📝 Reasoning Trace:")
        for i, step in enumerate(trace, 1):
            print(f"\nStep {i}:")
            print(f"  💭 Thought: {step.thought}")
            print(f"  🎯 Action: {step.action}")
            print(f"  📥 Input: {step.action_input}")
            print(f"  👁️ Observation: {step.observation[:100]}...")

if __name__ == "__main__":
    demonstrate_react()
