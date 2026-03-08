"""Main agent orchestrator using LangGraph's StateGraph for multi-agent workflows."""

from __future__ import annotations

import operator
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.base_agent import BaseAgent, create_llm
from src.agents.coder import CoderAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.reviewer import ReviewerAgent
from src.config import AgentConfig, LLMConfig, OrchestratorConfig
from src.memory.conversation import SharedMemory


class AgentState(TypedDict):
    """State schema for the multi-agent workflow graph."""

    messages: Annotated[list[BaseMessage], operator.add]
    next_agent: str
    task: str
    plan: str
    results: dict[str, str]
    iteration: int
    final_answer: str


class AgentType(str, Enum):
    """Available agent types in the orchestrator."""

    SUPERVISOR = "supervisor"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    CODER = "coder"
    REVIEWER = "reviewer"


class MultiAgentOrchestrator:
    """Orchestrates multiple specialized agents using LangGraph's StateGraph.

    The orchestrator implements a supervisor pattern where a central supervisor
    node routes tasks to specialized agents based on the current state and
    task requirements. Agents can operate sequentially or in parallel.

    Usage:
        config = OrchestratorConfig.from_env()
        orchestrator = MultiAgentOrchestrator(config)
        result = orchestrator.run("Build a REST API for user management")
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.config = config or OrchestratorConfig.from_env()
        self.memory = SharedMemory()
        self.agents: dict[str, BaseAgent] = {}
        self.supervisor_llm = create_llm(self.config.llm)

        self._init_agents()
        self.graph = self._build_graph()

    def _init_agents(self) -> None:
        """Initialize all specialized agents."""
        planner_config = self.config.agents.get(
            "planner", AgentConfig(name="planner", llm=self.config.llm)
        )
        researcher_config = self.config.agents.get(
            "researcher", AgentConfig(name="researcher", llm=self.config.llm)
        )
        coder_config = self.config.agents.get(
            "coder", AgentConfig(name="coder", llm=self.config.llm)
        )
        reviewer_config = self.config.agents.get(
            "reviewer", AgentConfig(name="reviewer", llm=self.config.llm)
        )

        self.agents["planner"] = PlannerAgent(config=planner_config)
        self.agents["researcher"] = ResearcherAgent(config=researcher_config)
        self.agents["coder"] = CoderAgent(config=coder_config)
        self.agents["reviewer"] = ReviewerAgent(config=reviewer_config)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph for the multi-agent workflow.

        Returns:
            A compiled StateGraph ready for execution.
        """
        workflow = StateGraph(AgentState)

        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("planner", self._agent_node("planner"))
        workflow.add_node("researcher", self._agent_node("researcher"))
        workflow.add_node("coder", self._agent_node("coder"))
        workflow.add_node("reviewer", self._agent_node("reviewer"))

        workflow.set_entry_point("supervisor")

        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "planner": "planner",
                "researcher": "researcher",
                "coder": "coder",
                "reviewer": "reviewer",
                "FINISH": END,
            },
        )

        for agent_name in ["planner", "researcher", "coder", "reviewer"]:
            workflow.add_edge(agent_name, "supervisor")

        return workflow.compile()

    def _supervisor_node(self, state: AgentState) -> dict[str, Any]:
        """Supervisor node that decides which agent should handle the next step.

        Args:
            state: Current workflow state.

        Returns:
            Updated state with routing decision.
        """
        context = self.memory.get_context_string(n=20)
        results_summary = "\n".join(
            f"- {k}: {v[:200]}" for k, v in state.get("results", {}).items()
        )

        supervisor_prompt = (
            "You are a supervisor managing a team of specialized agents:\n"
            "- planner: Decomposes complex tasks into steps\n"
            "- researcher: Gathers information via web search and knowledge base\n"
            "- coder: Writes and executes code\n"
            "- reviewer: Reviews code quality and provides feedback\n\n"
            f"Original task: {state.get('task', '')}\n\n"
            f"Current plan: {state.get('plan', 'No plan yet')}\n\n"
            f"Results so far:\n{results_summary or 'None yet'}\n\n"
            f"Recent context:\n{context}\n\n"
            f"Current iteration: {state.get('iteration', 0)}\n\n"
            "Based on the current state, decide the next action.\n"
            "Respond with ONLY one of: planner, researcher, coder, reviewer, FINISH\n"
            "Choose FINISH when the task is complete and a final answer can be provided."
        )

        messages = [
            SystemMessage(content=supervisor_prompt),
            HumanMessage(content=f"What should be the next step for: {state.get('task', '')}"),
        ]

        response = self.supervisor_llm.invoke(messages)
        next_agent = response.content.strip().lower()  # type: ignore[union-attr]

        valid_agents = {"planner", "researcher", "coder", "reviewer", "finish"}
        if next_agent not in valid_agents:
            next_agent = "planner"

        iteration = state.get("iteration", 0) + 1
        if iteration >= self.config.recursion_limit:
            next_agent = "finish"

        return {
            "next_agent": next_agent,
            "iteration": iteration,
            "messages": [AIMessage(content=f"Supervisor: routing to {next_agent}")],
        }

    def _agent_node(self, agent_name: str):
        """Create a node function for a specific agent.

        Args:
            agent_name: Name of the agent to create a node for.

        Returns:
            A callable node function for the StateGraph.
        """
        def node_fn(state: AgentState) -> dict[str, Any]:
            agent = self.agents[agent_name]
            context = self.memory.get_context_string(n=10)

            task_message = HumanMessage(
                content=(
                    f"Task: {state.get('task', '')}\n\n"
                    f"Plan: {state.get('plan', 'No plan yet')}\n\n"
                    f"Context from other agents:\n{context}\n\n"
                    "Please complete your part of this task."
                )
            )

            response = agent.invoke([task_message])
            response_text = response.content if isinstance(response.content, str) else str(response.content)

            self.memory.add_entry(
                agent_name=agent_name,
                content=response_text[:500],
                entry_type="result",
            )

            results = dict(state.get("results", {}))
            results[agent_name] = response_text

            update: dict[str, Any] = {
                "messages": [response],
                "results": results,
            }

            if agent_name == "planner":
                update["plan"] = response_text

            return update

        return node_fn

    def _route_from_supervisor(self, state: AgentState) -> str:
        """Routing function for the supervisor's conditional edges.

        Args:
            state: Current workflow state.

        Returns:
            The name of the next node to execute.
        """
        next_agent = state.get("next_agent", "planner")
        if next_agent == "finish":
            return "FINISH"
        return next_agent

    def run(self, task: str) -> str:
        """Execute the multi-agent workflow for a given task.

        Args:
            task: The task description to process.

        Returns:
            The final result from the workflow.
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "next_agent": "supervisor",
            "task": task,
            "plan": "",
            "results": {},
            "iteration": 0,
            "final_answer": "",
        }

        final_state = self.graph.invoke(
            initial_state,
            config={"recursion_limit": self.config.recursion_limit},
        )

        results = final_state.get("results", {})
        if results:
            last_key = list(results.keys())[-1]
            return results[last_key]

        messages = final_state.get("messages", [])
        if messages:
            return messages[-1].content if isinstance(messages[-1].content, str) else str(messages[-1].content)

        return "No result produced."

    async def arun(self, task: str) -> str:
        """Async version of run.

        Args:
            task: The task description to process.

        Returns:
            The final result from the workflow.
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=task)],
            "next_agent": "supervisor",
            "task": task,
            "plan": "",
            "results": {},
            "iteration": 0,
            "final_answer": "",
        }

        final_state = await self.graph.ainvoke(
            initial_state,
            config={"recursion_limit": self.config.recursion_limit},
        )

        results = final_state.get("results", {})
        if results:
            last_key = list(results.keys())[-1]
            return results[last_key]

        return "No result produced."
