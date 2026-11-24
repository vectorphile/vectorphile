"""vectorphile.agents — LLM Agent Framework.

Provides:
- Tool definition and registry
- ReAct agent loop (Reason + Act)
- Chain-of-Thought prompting
- Memory (in-context, vector store stub)
- Multi-agent orchestration

No external dependencies beyond vectorphile.

Usage::

    from vectorphile.agents import Agent, Tool, ReActAgent

    @Tool.register(name='search', description='Search for information')
    def search(query: str) -> str:
        return f'Result for: {query}'

    agent = ReActAgent(tools=[search], max_steps=10)
    result = agent.run('What is the capital of France?')
"""
from __future__ import annotations
import re
import json
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

class Tool:
    """A callable tool with a name, description, and schema.

    Usage::

        def search(query: str) -> str:
            return f'Results for {query}'

        tool = Tool(name='search', fn=search,
                    description='Search for information.',
                    parameters={'query': {'type': 'string'}})
        result = tool('What is 2+2?')
    """

    _registry: Dict[str, 'Tool'] = {}

    def __init__(self, name: str, fn: Callable, description: str = '',
                 parameters: Optional[Dict] = None):
        self.name = name
        self.fn = fn
        self.description = description
        self.parameters = parameters or {}

    def __call__(self, *args, **kwargs) -> Any:
        return self.fn(*args, **kwargs)

    @classmethod
    def register(cls, name: str, description: str = '',
                 parameters: Optional[Dict] = None):
        """Decorator: register a function as a named Tool."""
        def _decorator(fn: Callable) -> 'Tool':
            tool = cls(name=name, fn=fn, description=description,
                       parameters=parameters or {})
            cls._registry[name] = tool
            return tool
        return _decorator

    @classmethod
    def get(cls, name: str) -> Optional['Tool']:
        return cls._registry.get(name)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        }

    def __repr__(self) -> str:
        return f'Tool(name={self.name!r})'


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class ConversationMemory:
    """Simple in-context conversation memory (list of messages).

    Usage::

        mem = ConversationMemory(max_turns=10)
        mem.add('user', 'Hello')
        mem.add('assistant', 'Hi there!')
        messages = mem.messages()
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._history: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._history.append({'role': role, 'content': content})
        if len(self._history) > self.max_turns * 2:
            # Keep system + latest messages
            self._history = self._history[-self.max_turns * 2:]

    def messages(self) -> List[Dict[str, str]]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)


class SummaryMemory:
    """Summarizing memory: compresses old turns into a rolling summary."""

    def __init__(self, max_recent: int = 6):
        self.max_recent = max_recent
        self._summary = ''
        self._recent: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self._recent.append({'role': role, 'content': content})
        if len(self._recent) > self.max_recent * 2:
            # Compress old turns into summary
            old = self._recent[:self.max_recent]
            self._summary += ' '.join(f"{m['role']}: {m['content'][:100]}" for m in old) + ' '
            self._recent = self._recent[self.max_recent:]

    def messages(self) -> List[Dict[str, str]]:
        result = []
        if self._summary:
            result.append({'role': 'system',
                           'content': f'[Conversation summary]: {self._summary}'})
        result.extend(self._recent)
        return result

    def clear(self) -> None:
        self._summary = ''
        self._recent.clear()


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class AgentResult:
    """Result from an agent run."""

    def __init__(self, output: str, steps: List[Dict], metadata: Dict = None):
        self.output = output
        self.steps = steps
        self.metadata = metadata or {}

    def __str__(self) -> str:
        return self.output

    def __repr__(self) -> str:
        return f'AgentResult(output={self.output[:80]!r}, steps={len(self.steps)})'


class BaseAgent:
    """Abstract base agent."""

    def __init__(self, tools: List[Tool] = None, memory: Optional[ConversationMemory] = None,
                 max_steps: int = 10, verbose: bool = False):
        self.tools: Dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.memory = memory or ConversationMemory()
        self.max_steps = max_steps
        self.verbose = verbose

    def add_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def run(self, query: str) -> AgentResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ReAct Agent (Reason + Act)
# ---------------------------------------------------------------------------

class ReActAgent(BaseAgent):
    """ReAct agent: interleave Thought → Action → Observation loops.

    Parses model output for Thought/Action/Action Input/Observation markers
    and runs tool calls in a loop until Final Answer is found.

    Usage::

        @Tool.register('calculator', description='Evaluate math')
        def calc(expr): return str(eval(expr))

        agent = ReActAgent(tools=[calc], llm=my_llm_fn)
        result = agent.run('What is 2 + 2?')
    """

    THOUGHT_RE = re.compile(r'Thought:\s*(.*?)(?=Action:|Final Answer:|$)', re.DOTALL)
    ACTION_RE = re.compile(r'Action:\s*(\w+)', re.DOTALL)
    INPUT_RE = re.compile(r'Action Input:\s*(.*?)(?=Observation:|Thought:|Final Answer:|$)', re.DOTALL)
    FINAL_RE = re.compile(r'Final Answer:\s*(.*)', re.DOTALL)

    def __init__(self, tools: List[Tool] = None, llm: Optional[Callable] = None,
                 memory: Optional[ConversationMemory] = None,
                 max_steps: int = 10, verbose: bool = False):
        super().__init__(tools=tools, memory=memory, max_steps=max_steps, verbose=verbose)
        self._llm = llm  # llm(prompt: str) -> str

    def _build_prompt(self, query: str, scratchpad: str) -> str:
        tools_desc = '\n'.join(
            f'{t.name}: {t.description}' for t in self.tools.values()
        )
        tool_names = ', '.join(self.tools.keys())
        return (
            f'You are a helpful assistant with access to tools.\n\n'
            f'Tools available:\n{tools_desc}\n\n'
            f'Use this format:\n'
            f'Thought: <your reasoning>\n'
            f'Action: <tool name from [{tool_names}]>\n'
            f'Action Input: <tool input>\n'
            f'Observation: <result>\n'
            f'... (repeat Thought/Action/Observation as needed)\n'
            f'Final Answer: <your final answer>\n\n'
            f'Question: {query}\n'
            f'{scratchpad}'
        )

    def _parse_action(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        action_m = self.ACTION_RE.search(text)
        input_m = self.INPUT_RE.search(text)
        action = action_m.group(1).strip() if action_m else None
        action_input = input_m.group(1).strip() if input_m else ''
        return action, action_input

    def _run_tool(self, name: str, input_str: str) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f'Error: tool {name!r} not found.'
        try:
            # Try JSON parse first, fall back to raw string
            try:
                args = json.loads(input_str)
                if isinstance(args, dict):
                    result = tool(**args)
                else:
                    result = tool(args)
            except (json.JSONDecodeError, TypeError):
                result = tool(input_str)
            return str(result)
        except Exception as e:
            return f'Error: {e}'

    def run(self, query: str) -> AgentResult:
        scratchpad = ''
        steps = []
        for step in range(self.max_steps):
            prompt = self._build_prompt(query, scratchpad)
            if self._llm is None:
                # No LLM: just return a stub
                return AgentResult(
                    output=f'[No LLM configured] Query: {query}',
                    steps=steps,
                )
            response = self._llm(prompt)
            if self.verbose:
                print(f'[Step {step+1}] {response[:200]}')

            # Check for final answer
            final_m = self.FINAL_RE.search(response)
            if final_m:
                answer = final_m.group(1).strip()
                steps.append({'type': 'final', 'answer': answer})
                return AgentResult(output=answer, steps=steps)

            # Parse and execute action
            action, action_input = self._parse_action(response)
            if action:
                observation = self._run_tool(action, action_input)
                scratchpad += f'{response}\nObservation: {observation}\n'
                steps.append({
                    'type': 'action',
                    'thought': (self.THOUGHT_RE.search(response) or type('', (), {'group': lambda s, x: ''})()).group(1),
                    'action': action,
                    'input': action_input,
                    'observation': observation,
                })
            else:
                # No parseable action — treat as final
                return AgentResult(output=response.strip(), steps=steps)

        return AgentResult(
            output=f'Reached max steps ({self.max_steps}). Last scratchpad: {scratchpad[-200:]}',
            steps=steps,
        )


# ---------------------------------------------------------------------------
# Chain-of-Thought Agent
# ---------------------------------------------------------------------------

class ChainOfThoughtAgent(BaseAgent):
    """Chain-of-Thought agent: appends 'Let's think step by step' and parses.

    Usage::

        agent = ChainOfThoughtAgent(llm=my_llm)
        result = agent.run('Solve: a train travels 60mph for 2 hours...')
    """

    def __init__(self, llm: Callable, tools: List[Tool] = None, **kwargs):
        super().__init__(tools=tools, **kwargs)
        self._llm = llm

    def run(self, query: str) -> AgentResult:
        prompt = (
            f'{query}\n\n'
            f"Let's think step by step:"
        )
        response = self._llm(prompt)
        # Extract final answer (last sentence or after "Therefore")
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        output = lines[-1] if lines else response.strip()
        return AgentResult(output=output, steps=[{'type': 'cot', 'reasoning': response}])


# ---------------------------------------------------------------------------
# Multi-agent orchestration
# ---------------------------------------------------------------------------

class AgentPipeline:
    """Run agents in sequence, passing output of each as input to next.

    Usage::

        pipeline = AgentPipeline([researcher, writer, editor])
        result = pipeline.run('Write a report on climate change')
    """

    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents

    def run(self, query: str) -> AgentResult:
        current_input = query
        all_steps = []
        for i, agent in enumerate(self.agents):
            result = agent.run(current_input)
            all_steps.extend(result.steps)
            current_input = result.output
        return AgentResult(output=current_input, steps=all_steps)


class ParallelAgents:
    """Run agents in parallel (sequential fallback; real parallelism optional).

    Usage::

        agents = ParallelAgents([agent1, agent2, agent3])
        results = agents.run('Analyze this dataset from different angles')
    """

    def __init__(self, agents: List[BaseAgent], aggregator: Optional[Callable] = None):
        self.agents = agents
        self._aggregator = aggregator or (lambda results: '\n\n'.join(str(r) for r in results))

    def run(self, query: str) -> AgentResult:
        results = []
        all_steps = []
        for agent in self.agents:
            r = agent.run(query)
            results.append(r.output)
            all_steps.extend(r.steps)
        aggregated = self._aggregator(results)
        return AgentResult(output=aggregated, steps=all_steps)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

class PromptTemplate:
    """Simple prompt template with variable substitution.

    Usage::

        tmpl = PromptTemplate('Translate {text} to {language}.')
        prompt = tmpl.format(text='Hello', language='French')
    """

    def __init__(self, template: str):
        self.template = template
        # Extract variable names
        self.variables = re.findall(r'\{(\w+)\}', template)

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

    def __repr__(self) -> str:
        return f'PromptTemplate(vars={self.variables})'


class FewShotPrompt:
    """Few-shot prompt builder.

    Usage::

        fs = FewShotPrompt(
            prefix='Classify sentiment:',
            examples=[('I love it', 'positive'), ('Terrible', 'negative')],
            suffix='Input: {input}\nOutput:',
        )
        prompt = fs.format(input='This is amazing!')
    """

    def __init__(self, prefix: str, examples: List[Tuple[str, str]],
                 suffix: str, example_separator: str = '\n\n'):
        self.prefix = prefix
        self.examples = examples
        self.suffix = suffix
        self.sep = example_separator

    def format(self, **kwargs) -> str:
        examples_str = self.sep.join(
            f'Input: {inp}\nOutput: {out}' for inp, out in self.examples
        )
        suffix = self.suffix.format(**kwargs)
        return f'{self.prefix}\n\n{examples_str}\n\n{suffix}'


__all__ = [
    'Tool', 'ConversationMemory', 'SummaryMemory',
    'AgentResult', 'BaseAgent', 'ReActAgent', 'ChainOfThoughtAgent',
    'AgentPipeline', 'ParallelAgents',
    'PromptTemplate', 'FewShotPrompt',
]
