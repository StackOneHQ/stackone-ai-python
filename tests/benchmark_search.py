"""
Benchmark comparing local BM25+TF-IDF vs semantic search.

Expected results:
- Local BM25+TF-IDF: ~21% Hit@5
- Semantic Search: ~84% Hit@5
- Improvement: 4x

Run with production API:
    STACKONE_API_KEY=xxx python tests/benchmark_search.py

Run with local Lambda (ai-generation/apps/action_search):
    # First, start the local Lambda:
    #   cd ai-generation/apps/action_search && make run-local
    # Then run benchmark:
    python tests/benchmark_search.py --local

Environment Variables:
    STACKONE_API_KEY: Required for production mode
    LOCAL_LAMBDA_URL: Optional, defaults to http://localhost:4513/2015-03-31/functions/function/invocations
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx

from stackone_ai.semantic_search import SemanticSearchClient, SemanticSearchResponse, SemanticSearchResult
from stackone_ai.utility_tools import ToolIndex

# Default local Lambda URL (from ai-generation/apps/action_search docker-compose)
DEFAULT_LOCAL_LAMBDA_URL = "http://localhost:4513/2015-03-31/functions/function/invocations"


class SearchClientProtocol(Protocol):
    """Protocol for search clients (production or local)."""

    def search(
        self,
        query: str,
        connector: str | None = None,
        top_k: int = 10,
    ) -> SemanticSearchResponse: ...


class LocalLambdaSearchClient:
    """Client for local action_search Lambda.

    This client connects to the local Lambda running via docker-compose
    from ai-generation/apps/action_search.

    Usage:
        # Start local Lambda first:
        #   cd ai-generation/apps/action_search && make run-local

        client = LocalLambdaSearchClient()
        response = client.search("create employee", connector="bamboohr", top_k=5)
    """

    def __init__(
        self,
        lambda_url: str = DEFAULT_LOCAL_LAMBDA_URL,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the local Lambda client.

        Args:
            lambda_url: URL of the local Lambda endpoint
            timeout: Request timeout in seconds
        """
        self.lambda_url = lambda_url
        self.timeout = timeout

    def _invoke(self, event: dict[str, Any]) -> dict[str, Any]:
        """Invoke the local Lambda with an event payload."""
        response = httpx.post(
            self.lambda_url,
            json=event,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def _parse_results(self, data: dict[str, Any], query: str) -> SemanticSearchResponse:
        """Parse Lambda response into SemanticSearchResponse."""
        results = [
            SemanticSearchResult(
                action_name=r.get("action_name", ""),
                connector_key=r.get("connector_key", ""),
                similarity_score=r.get("similarity_score", 0.0),
                label=r.get("label", ""),
                description=r.get("description", ""),
            )
            for r in data.get("results", [])
        ]
        return SemanticSearchResponse(
            results=results,
            total_count=data.get("total_count", len(results)),
            query=data.get("query", query),
        )

    def search(
        self,
        query: str,
        connector: str | None = None,
        top_k: int = 10,
    ) -> SemanticSearchResponse:
        """Search for relevant actions using local Lambda.

        Args:
            query: Natural language query
            connector: Optional connector filter
            top_k: Maximum number of results

        Returns:
            SemanticSearchResponse with matching actions
        """
        payload: dict[str, Any] = {
            "type": "search",
            "payload": {"query": query, "top_k": top_k},
        }
        if connector:
            payload["payload"]["connector"] = connector

        try:
            data = self._invoke(payload)
            return self._parse_results(data, query)
        except httpx.RequestError as e:
            raise RuntimeError(f"Local Lambda request failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Local Lambda search failed: {e}") from e

    def fetch_all_actions(self) -> list[SemanticSearchResult]:
        """Fetch a broad set of actions from the Lambda for building local BM25 index.

        Uses multiple broad queries with high top_k to collect the full action catalog.
        This avoids needing the /mcp endpoint or STACKONE_API_KEY for benchmarking.

        Returns:
            Deduplicated list of all available actions
        """
        broad_queries = [
            "employee",
            "candidate",
            "contact",
            "task",
            "message",
            "file",
            "user",
            "event",
            "campaign",
            "course",
            "deal",
            "account",
            "job",
            "interview",
            "department",
            "time off",
            "comment",
            "project",
            "folder",
            "role",
        ]

        seen: dict[str, SemanticSearchResult] = {}
        for query in broad_queries:
            try:
                data = self._invoke(
                    {
                        "type": "search",
                        "payload": {"query": query, "top_k": 500},
                    }
                )
                for r in data.get("results", []):
                    name = r.get("action_name", "")
                    if name and name not in seen:
                        seen[name] = SemanticSearchResult(
                            action_name=name,
                            connector_key=r.get("connector_key", ""),
                            similarity_score=r.get("similarity_score", 0.0),
                            label=r.get("label", ""),
                            description=r.get("description", ""),
                        )
            except Exception:
                continue

        return list(seen.values())


@dataclass
class LightweightTool:
    """Minimal tool representation for BM25 indexing (no API dependency)."""

    name: str
    description: str


@dataclass
class EvaluationTask:
    """Single evaluation task for benchmark."""

    id: str
    query: str
    category: str
    complexity: Literal["simple", "moderate", "complex"]
    expected_matches: list[str]
    connector: str | None = None


# 103 semantically-challenging evaluation queries
# Ported from ai-generation/apps/action_search/tests/benchmark.integration.spec.ts
EVALUATION_TASKS: list[EvaluationTask] = [
    # ============ ALL CONNECTORS - SEMANTIC CHALLENGES ============
    # HR/HRIS - Natural language
    EvaluationTask(
        id="hr-sem-01",
        query="onboard a new team member",
        category="hr",
        complexity="moderate",
        expected_matches=["Create Employee", "Add Employee", "employee"],
    ),
    EvaluationTask(
        id="hr-sem-02",
        query="fetch staff information",
        category="hr",
        complexity="simple",
        expected_matches=["Get Employee", "Get Worker", "List Employees", "employee", "worker"],
    ),
    EvaluationTask(
        id="hr-sem-03",
        query="request vacation days",
        category="hr",
        complexity="moderate",
        expected_matches=["Create Time Off", "Create Absence", "Time-Off", "absence", "leave"],
    ),
    EvaluationTask(
        id="hr-sem-04",
        query="show me everyone in the company",
        category="hr",
        complexity="simple",
        expected_matches=["List Employees", "List Workers", "employees", "workers"],
    ),
    EvaluationTask(
        id="hr-sem-05",
        query="change someone's job title",
        category="hr",
        complexity="moderate",
        expected_matches=["Update Employee", "Job Change", "Update Worker", "employee"],
    ),
    EvaluationTask(
        id="hr-sem-06",
        query="terminate an employee",
        category="hr",
        complexity="moderate",
        expected_matches=["Delete Employee", "Terminate", "employee"],
    ),
    EvaluationTask(
        id="hr-sem-07",
        query="pull the org chart",
        category="hr",
        complexity="moderate",
        expected_matches=["List Departments", "Organization", "hierarchy", "departments"],
    ),
    EvaluationTask(
        id="hr-sem-08",
        query="sick day request",
        category="hr",
        complexity="simple",
        expected_matches=["Create Absence", "Time-Off", "Leave", "absence"],
    ),
    EvaluationTask(
        id="hr-sem-09",
        query="get employee details",
        category="hr",
        complexity="simple",
        expected_matches=["Get Employee", "employee"],
    ),
    EvaluationTask(
        id="hr-sem-10",
        query="update staff record",
        category="hr",
        complexity="simple",
        expected_matches=["Update Employee", "employee"],
    ),
    EvaluationTask(
        id="hr-sem-11",
        query="add new hire to the system",
        category="hr",
        complexity="moderate",
        expected_matches=["Create Employee", "employee"],
    ),
    EvaluationTask(
        id="hr-sem-12",
        query="who works in engineering",
        category="hr",
        complexity="moderate",
        expected_matches=["List Employees", "employees", "department"],
    ),
    EvaluationTask(
        id="hr-sem-13",
        query="view compensation details",
        category="hr",
        complexity="moderate",
        expected_matches=["Get Employee", "compensation", "salary"],
    ),
    EvaluationTask(
        id="hr-sem-14",
        query="see all time-off requests",
        category="hr",
        complexity="simple",
        expected_matches=["List Time Off", "List Absences", "time_off", "absence"],
    ),
    EvaluationTask(
        id="hr-sem-15",
        query="approve PTO",
        category="hr",
        complexity="moderate",
        expected_matches=["Update Time Off", "Update Absence", "time_off", "absence"],
    ),
    # Recruiting/ATS - Natural language
    EvaluationTask(
        id="ats-sem-01",
        query="bring in a new applicant",
        category="recruiting",
        complexity="moderate",
        expected_matches=["Create Candidate", "Create Application", "candidate", "application"],
    ),
    EvaluationTask(
        id="ats-sem-02",
        query="see who applied for the role",
        category="recruiting",
        complexity="simple",
        expected_matches=["List Candidates", "List Applications", "candidates", "applications"],
    ),
    EvaluationTask(
        id="ats-sem-03",
        query="advance someone to the next round",
        category="recruiting",
        complexity="moderate",
        expected_matches=["Move Application", "Update Stage", "stage", "move"],
    ),
    EvaluationTask(
        id="ats-sem-04",
        query="turn down a job seeker",
        category="recruiting",
        complexity="moderate",
        expected_matches=["Reject", "Disqualify", "reject", "application"],
    ),
    EvaluationTask(
        id="ats-sem-05",
        query="post a new position",
        category="recruiting",
        complexity="simple",
        expected_matches=["Create Job", "Job Posting", "job"],
    ),
    EvaluationTask(
        id="ats-sem-06",
        query="schedule an interview",
        category="recruiting",
        complexity="moderate",
        expected_matches=["Create Interview", "Schedule", "interview"],
    ),
    EvaluationTask(
        id="ats-sem-07",
        query="view candidate resume",
        category="recruiting",
        complexity="simple",
        expected_matches=["Get Candidate", "candidate", "document"],
    ),
    EvaluationTask(
        id="ats-sem-08",
        query="add interview feedback",
        category="recruiting",
        complexity="moderate",
        expected_matches=["Create Scorecard", "scorecard", "feedback"],
    ),
    EvaluationTask(
        id="ats-sem-09",
        query="check application status",
        category="recruiting",
        complexity="simple",
        expected_matches=["Get Application", "application"],
    ),
    EvaluationTask(
        id="ats-sem-10",
        query="see open positions",
        category="recruiting",
        complexity="simple",
        expected_matches=["List Jobs", "jobs"],
    ),
    # CRM - Natural language
    EvaluationTask(
        id="crm-sem-01",
        query="add a new prospect",
        category="crm",
        complexity="simple",
        expected_matches=["Create Lead", "Create Contact", "lead", "contact"],
    ),
    EvaluationTask(
        id="crm-sem-02",
        query="log a sales opportunity",
        category="crm",
        complexity="moderate",
        expected_matches=["Create Deal", "Create Opportunity", "deal", "opportunity"],
    ),
    EvaluationTask(
        id="crm-sem-03",
        query="close a deal",
        category="crm",
        complexity="moderate",
        expected_matches=["Update Deal", "Update Opportunity", "deal", "opportunity"],
    ),
    EvaluationTask(
        id="crm-sem-04",
        query="find customer information",
        category="crm",
        complexity="simple",
        expected_matches=["Get Contact", "Get Account", "contact", "account"],
    ),
    EvaluationTask(
        id="crm-sem-05",
        query="create a new account",
        category="crm",
        complexity="simple",
        expected_matches=["Create Account", "account"],
    ),
    EvaluationTask(
        id="crm-sem-06",
        query="log a sales call",
        category="crm",
        complexity="moderate",
        expected_matches=["Create Activity", "activity", "call"],
    ),
    EvaluationTask(
        id="crm-sem-07",
        query="see pipeline deals",
        category="crm",
        complexity="simple",
        expected_matches=["List Deals", "List Opportunities", "deals", "opportunities"],
    ),
    EvaluationTask(
        id="crm-sem-08",
        query="update contact info",
        category="crm",
        complexity="simple",
        expected_matches=["Update Contact", "contact"],
    ),
    EvaluationTask(
        id="crm-sem-09",
        query="track customer interaction",
        category="crm",
        complexity="moderate",
        expected_matches=["Create Activity", "activity"],
    ),
    EvaluationTask(
        id="crm-sem-10",
        query="view all contacts",
        category="crm",
        complexity="simple",
        expected_matches=["List Contacts", "contacts"],
    ),
    # Project Management - Natural language
    EvaluationTask(
        id="pm-sem-01",
        query="assign work to someone",
        category="project",
        complexity="simple",
        expected_matches=["Create Task", "Create Issue", "Assign", "task", "issue"],
    ),
    EvaluationTask(
        id="pm-sem-02",
        query="check my to-do list",
        category="project",
        complexity="simple",
        expected_matches=["List Tasks", "List Issues", "tasks", "issues"],
    ),
    EvaluationTask(
        id="pm-sem-03",
        query="file a bug report",
        category="project",
        complexity="moderate",
        expected_matches=["Create Issue", "Create Task", "issue"],
    ),
    EvaluationTask(
        id="pm-sem-04",
        query="mark task as done",
        category="project",
        complexity="simple",
        expected_matches=["Update Task", "Update Issue", "task", "issue"],
    ),
    EvaluationTask(
        id="pm-sem-05",
        query="create a new project",
        category="project",
        complexity="simple",
        expected_matches=["Create Project", "project"],
    ),
    EvaluationTask(
        id="pm-sem-06",
        query="view project status",
        category="project",
        complexity="simple",
        expected_matches=["Get Project", "project"],
    ),
    EvaluationTask(
        id="pm-sem-07",
        query="add a comment to ticket",
        category="project",
        complexity="moderate",
        expected_matches=["Create Comment", "comment"],
    ),
    EvaluationTask(
        id="pm-sem-08",
        query="see sprint backlog",
        category="project",
        complexity="moderate",
        expected_matches=["List Tasks", "List Issues", "tasks", "issues"],
    ),
    # Messaging - Natural language
    EvaluationTask(
        id="msg-sem-01",
        query="ping my colleague",
        category="messaging",
        complexity="simple",
        expected_matches=["Send Message", "message"],
    ),
    EvaluationTask(
        id="msg-sem-02",
        query="start a group chat",
        category="messaging",
        complexity="moderate",
        expected_matches=["Create Conversation", "Create Channel", "conversation", "channel"],
    ),
    EvaluationTask(
        id="msg-sem-03",
        query="post in the team channel",
        category="messaging",
        complexity="simple",
        expected_matches=["Send Message", "message", "channel"],
    ),
    EvaluationTask(
        id="msg-sem-04",
        query="see recent messages",
        category="messaging",
        complexity="simple",
        expected_matches=["List Messages", "messages"],
    ),
    EvaluationTask(
        id="msg-sem-05",
        query="create a new channel",
        category="messaging",
        complexity="simple",
        expected_matches=["Create Channel", "channel"],
    ),
    # Documents - Natural language
    EvaluationTask(
        id="doc-sem-01",
        query="upload a file",
        category="documents",
        complexity="simple",
        expected_matches=["Upload File", "Create File", "file", "upload"],
    ),
    EvaluationTask(
        id="doc-sem-02",
        query="download the document",
        category="documents",
        complexity="simple",
        expected_matches=["Download File", "Get File", "file", "download"],
    ),
    EvaluationTask(
        id="doc-sem-03",
        query="see all shared files",
        category="documents",
        complexity="simple",
        expected_matches=["List Files", "files"],
    ),
    EvaluationTask(
        id="doc-sem-04",
        query="create a new folder",
        category="documents",
        complexity="simple",
        expected_matches=["Create Folder", "folder"],
    ),
    EvaluationTask(
        id="doc-sem-05",
        query="share document with team",
        category="documents",
        complexity="moderate",
        expected_matches=["Share File", "Update File", "file", "share"],
    ),
    # Marketing - Natural language
    EvaluationTask(
        id="mkt-sem-01",
        query="create email campaign",
        category="marketing",
        complexity="moderate",
        expected_matches=["Create Campaign", "campaign", "email"],
    ),
    EvaluationTask(
        id="mkt-sem-02",
        query="add contact to mailing list",
        category="marketing",
        complexity="simple",
        expected_matches=["Add Member", "Create Contact", "contact", "list"],
    ),
    EvaluationTask(
        id="mkt-sem-03",
        query="send newsletter",
        category="marketing",
        complexity="moderate",
        expected_matches=["Send Campaign", "campaign", "email"],
    ),
    EvaluationTask(
        id="mkt-sem-04",
        query="view campaign analytics",
        category="marketing",
        complexity="moderate",
        expected_matches=["Get Campaign", "campaign", "analytics"],
    ),
    EvaluationTask(
        id="mkt-sem-05",
        query="create automation workflow",
        category="marketing",
        complexity="complex",
        expected_matches=["Create Automation", "automation", "workflow"],
    ),
    # LMS - Natural language
    EvaluationTask(
        id="lms-sem-01",
        query="assign training to employee",
        category="lms",
        complexity="moderate",
        expected_matches=["Create Assignment", "Assign Content", "assignment", "content"],
    ),
    EvaluationTask(
        id="lms-sem-02",
        query="check course completion",
        category="lms",
        complexity="simple",
        expected_matches=["Get Completion", "completion", "progress"],
    ),
    EvaluationTask(
        id="lms-sem-03",
        query="create new course",
        category="lms",
        complexity="moderate",
        expected_matches=["Create Content", "content", "course"],
    ),
    EvaluationTask(
        id="lms-sem-04",
        query="see available trainings",
        category="lms",
        complexity="simple",
        expected_matches=["List Content", "content", "courses"],
    ),
    EvaluationTask(
        id="lms-sem-05",
        query="track learning progress",
        category="lms",
        complexity="moderate",
        expected_matches=["Get Completion", "List Completions", "completion"],
    ),
    # Per-connector examples
    EvaluationTask(
        id="bamboo-sem-01",
        query="bring on a new hire",
        category="hr",
        complexity="moderate",
        connector="bamboohr",
        expected_matches=["Create Employee", "employee"],
    ),
    EvaluationTask(
        id="bamboo-sem-02",
        query="get employee time off balance",
        category="hr",
        complexity="simple",
        connector="bamboohr",
        expected_matches=["Get Time Off", "time_off", "balance"],
    ),
    EvaluationTask(
        id="slack-sem-01",
        query="ping the team",
        category="messaging",
        complexity="simple",
        connector="slack",
        expected_matches=["Send Message", "message"],
    ),
    EvaluationTask(
        id="slack-sem-02",
        query="create team workspace",
        category="messaging",
        complexity="moderate",
        connector="slack",
        expected_matches=["Create Channel", "channel"],
    ),
    EvaluationTask(
        id="jira-sem-01",
        query="file a new bug",
        category="project",
        complexity="simple",
        connector="jira",
        expected_matches=["Create Issue", "issue"],
    ),
    EvaluationTask(
        id="jira-sem-02",
        query="view sprint tasks",
        category="project",
        complexity="simple",
        connector="jira",
        expected_matches=["List Issues", "issues"],
    ),
    EvaluationTask(
        id="greenhouse-sem-01",
        query="add new job posting",
        category="recruiting",
        complexity="simple",
        connector="greenhouse",
        expected_matches=["Create Job", "job"],
    ),
    EvaluationTask(
        id="greenhouse-sem-02",
        query="move candidate forward",
        category="recruiting",
        complexity="moderate",
        connector="greenhouse",
        expected_matches=["Move Application", "Update Application", "application"],
    ),
    EvaluationTask(
        id="salesforce-sem-01",
        query="create sales opportunity",
        category="crm",
        complexity="simple",
        connector="salesforce",
        expected_matches=["Create Opportunity", "opportunity"],
    ),
    EvaluationTask(
        id="salesforce-sem-02",
        query="log customer call",
        category="crm",
        complexity="moderate",
        connector="salesforce",
        expected_matches=["Create Activity", "activity"],
    ),
    EvaluationTask(
        id="hubspot-sem-01",
        query="add new lead",
        category="crm",
        complexity="simple",
        connector="hubspot",
        expected_matches=["Create Contact", "contact"],
    ),
    EvaluationTask(
        id="hubspot-sem-02",
        query="track deal progress",
        category="crm",
        complexity="moderate",
        connector="hubspot",
        expected_matches=["Get Deal", "Update Deal", "deal"],
    ),
    # Complex multi-step queries
    EvaluationTask(
        id="complex-01",
        query="set up new employee with all required training",
        category="hr",
        complexity="complex",
        expected_matches=["Create Employee", "Create Assignment", "employee", "assignment"],
    ),
    EvaluationTask(
        id="complex-02",
        query="process job application and schedule interview",
        category="recruiting",
        complexity="complex",
        expected_matches=["Create Application", "Create Interview", "application", "interview"],
    ),
    EvaluationTask(
        id="complex-03",
        query="update deal and notify team",
        category="crm",
        complexity="complex",
        expected_matches=["Update Deal", "Send Message", "deal", "message"],
    ),
    EvaluationTask(
        id="complex-04",
        query="create project and assign initial tasks",
        category="project",
        complexity="complex",
        expected_matches=["Create Project", "Create Task", "project", "task"],
    ),
    # Edge cases - Abbreviations and slang
    EvaluationTask(
        id="edge-01",
        query="PTO request",
        category="hr",
        complexity="simple",
        expected_matches=["Create Time Off", "time_off", "absence"],
    ),
    EvaluationTask(
        id="edge-02",
        query="1:1 meeting",
        category="hr",
        complexity="moderate",
        expected_matches=["Create Event", "Create Meeting", "meeting"],
    ),
    EvaluationTask(
        id="edge-03",
        query="OOO",
        category="hr",
        complexity="simple",
        expected_matches=["Time Off", "Absence", "time_off", "absence"],
    ),
    EvaluationTask(
        id="edge-04",
        query="ASAP task",
        category="project",
        complexity="simple",
        expected_matches=["Create Task", "task"],
    ),
    EvaluationTask(
        id="edge-05",
        query="DM someone",
        category="messaging",
        complexity="simple",
        expected_matches=["Send Message", "message"],
    ),
    # Synonyms and alternative phrases
    EvaluationTask(
        id="syn-01",
        query="fire someone",
        category="hr",
        complexity="moderate",
        expected_matches=["Delete Employee", "Terminate", "employee"],
    ),
    EvaluationTask(
        id="syn-02",
        query="look up customer",
        category="crm",
        complexity="simple",
        expected_matches=["Get Contact", "Get Account", "contact", "account"],
    ),
    EvaluationTask(
        id="syn-03",
        query="grab the file",
        category="documents",
        complexity="simple",
        expected_matches=["Download File", "Get File", "file"],
    ),
    EvaluationTask(
        id="syn-04",
        query="sign up new user",
        category="hr",
        complexity="moderate",
        expected_matches=["Create Employee", "Create User", "employee", "user"],
    ),
    EvaluationTask(
        id="syn-05",
        query="kill the ticket",
        category="project",
        complexity="moderate",
        expected_matches=["Delete Issue", "Update Issue", "Close Issue", "issue"],
    ),
    # Business context queries
    EvaluationTask(
        id="biz-01",
        query="run payroll",
        category="hr",
        complexity="complex",
        expected_matches=["payroll", "compensation"],
    ),
    EvaluationTask(
        id="biz-02",
        query="close quarter books",
        category="crm",
        complexity="complex",
        expected_matches=["Update Deal", "deal", "opportunity"],
    ),
    EvaluationTask(
        id="biz-03",
        query="annual review",
        category="hr",
        complexity="moderate",
        expected_matches=["Review", "Performance", "employee"],
    ),
    EvaluationTask(
        id="biz-04",
        query="sprint planning",
        category="project",
        complexity="moderate",
        expected_matches=["Create Task", "List Tasks", "task", "issue"],
    ),
    EvaluationTask(
        id="biz-05",
        query="customer onboarding",
        category="crm",
        complexity="complex",
        expected_matches=["Create Account", "Create Contact", "account", "contact"],
    ),
]


@dataclass
class TaskResult:
    """Result of evaluating a single task."""

    task_id: str
    query: str
    hit: bool
    rank: int | None  # Position of first match, None if not found
    top_results: list[str]
    latency_ms: float


@dataclass
class BenchmarkResult:
    """Aggregated results from running benchmark."""

    method: str
    hit_at_k: float
    mean_reciprocal_rank: float
    avg_latency_ms: float
    total_tasks: int
    hits: int
    results: list[TaskResult] = field(default_factory=list)


@dataclass
class ComparisonReport:
    """Comparison between local and semantic search."""

    local_results: BenchmarkResult
    semantic_results: BenchmarkResult

    @property
    def improvement(self) -> float:
        """Percentage point improvement in Hit@k."""
        return self.semantic_results.hit_at_k - self.local_results.hit_at_k


def check_hit(result_names: list[str], expected_matches: list[str]) -> tuple[bool, int | None]:
    """Check if any expected match appears in results (case-insensitive partial match)."""
    for i, name in enumerate(result_names):
        name_lower = name.lower()
        for expected in expected_matches:
            if expected.lower() in name_lower:
                return True, i + 1
    return False, None


class SearchBenchmark:
    """Benchmark comparing local vs semantic search."""

    def __init__(
        self,
        tools: list,
        semantic_client: SearchClientProtocol,
    ):
        """Initialize benchmark with tools and search client.

        Args:
            tools: List of tool objects (StackOneTool or LightweightTool) with name + description
            semantic_client: Client for semantic search (production or local)
        """
        self.tools = tools
        # ToolIndex uses duck typing - only needs .name and .description
        self.local_index = ToolIndex(tools)  # type: ignore[arg-type]
        self.semantic_client = semantic_client

    def evaluate_local(
        self,
        tasks: list[EvaluationTask],
        k: int = 5,
    ) -> BenchmarkResult:
        """Run local BM25+TF-IDF search against benchmark tasks.

        Args:
            tasks: List of evaluation tasks
            k: Number of top results to consider (default: 5)

        Returns:
            BenchmarkResult with aggregated metrics
        """
        results: list[TaskResult] = []
        total_rr = 0.0

        for task in tasks:
            start = time.perf_counter()
            search_results = self.local_index.search(task.query, limit=k)
            latency = (time.perf_counter() - start) * 1000

            result_names = [r.name for r in search_results]
            hit, rank = check_hit(result_names, task.expected_matches)

            if hit and rank:
                total_rr += 1.0 / rank

            results.append(
                TaskResult(
                    task_id=task.id,
                    query=task.query,
                    hit=hit,
                    rank=rank,
                    top_results=result_names[:k],
                    latency_ms=latency,
                )
            )

        hits = sum(1 for r in results if r.hit)
        return BenchmarkResult(
            method="Local BM25+TF-IDF",
            hit_at_k=hits / len(tasks) if tasks else 0,
            mean_reciprocal_rank=total_rr / len(tasks) if tasks else 0,
            avg_latency_ms=sum(r.latency_ms for r in results) / len(results) if results else 0,
            total_tasks=len(tasks),
            hits=hits,
            results=results,
        )

    def evaluate_semantic(
        self,
        tasks: list[EvaluationTask],
        k: int = 5,
    ) -> BenchmarkResult:
        """Run semantic search against benchmark tasks.

        Args:
            tasks: List of evaluation tasks
            k: Number of top results to consider (default: 5)

        Returns:
            BenchmarkResult with aggregated metrics
        """
        results: list[TaskResult] = []
        total_rr = 0.0

        for task in tasks:
            start = time.perf_counter()
            response = self.semantic_client.search(
                query=task.query,
                connector=task.connector,
                top_k=k,
            )
            latency = (time.perf_counter() - start) * 1000

            result_names = [r.action_name for r in response.results]
            hit, rank = check_hit(result_names, task.expected_matches)

            if hit and rank:
                total_rr += 1.0 / rank

            results.append(
                TaskResult(
                    task_id=task.id,
                    query=task.query,
                    hit=hit,
                    rank=rank,
                    top_results=result_names[:k],
                    latency_ms=latency,
                )
            )

        hits = sum(1 for r in results if r.hit)
        return BenchmarkResult(
            method="Semantic Search",
            hit_at_k=hits / len(tasks) if tasks else 0,
            mean_reciprocal_rank=total_rr / len(tasks) if tasks else 0,
            avg_latency_ms=sum(r.latency_ms for r in results) / len(results) if results else 0,
            total_tasks=len(tasks),
            hits=hits,
            results=results,
        )

    def compare(self, tasks: list[EvaluationTask] | None = None, k: int = 5) -> ComparisonReport:
        """Compare both methods and generate report.

        Args:
            tasks: List of evaluation tasks (defaults to EVALUATION_TASKS)
            k: Number of top results to consider (default: 5)

        Returns:
            ComparisonReport with results from both methods
        """
        tasks = tasks or EVALUATION_TASKS
        local = self.evaluate_local(tasks, k)
        semantic = self.evaluate_semantic(tasks, k)
        return ComparisonReport(local_results=local, semantic_results=semantic)


def print_report(report: ComparisonReport) -> None:
    """Print a formatted comparison report."""
    print("\n" + "=" * 70)
    print("SEARCH BENCHMARK COMPARISON")
    print("=" * 70)

    print(f"\n{'Method':<25} {'Hit@5':<12} {'MRR':<12} {'Latency':<12} {'Hits':<10}")
    print("-" * 70)

    for r in [report.local_results, report.semantic_results]:
        print(
            f"{r.method:<25} {r.hit_at_k:>10.1%} {r.mean_reciprocal_rank:>10.3f} "
            f"{r.avg_latency_ms:>9.1f}ms {r.hits:>4}/{r.total_tasks}"
        )

    print("-" * 70)
    print(f"{'Improvement':<25} {report.improvement:>+10.1%}")
    print("=" * 70)

    # Build lookup maps
    local_by_id = {r.task_id: r for r in report.local_results.results}
    semantic_by_id = {r.task_id: r for r in report.semantic_results.results}

    failed_local = [r for r in report.local_results.results if not r.hit]
    failed_semantic = [r for r in report.semantic_results.results if not r.hit]

    # Tasks semantic gets right but local misses (the value semantic adds)
    semantic_wins = [r for r in failed_local if semantic_by_id.get(r.task_id, r).hit]
    # Tasks local gets right but semantic misses
    local_wins = [r for r in failed_semantic if local_by_id.get(r.task_id, r).hit]
    # Tasks both miss
    both_miss = [r for r in failed_local if not semantic_by_id.get(r.task_id, r).hit]

    print(f"\n{'SEMANTIC WINS':} ({len(semantic_wins)} tasks - semantic gets right, local misses):")
    for r in semantic_wins:
        sr = semantic_by_id[r.task_id]
        print(f"  - {r.task_id}: '{r.query}'")
        print(f"    Local got:    {r.top_results[:3]}")
        print(f"    Semantic got: {sr.top_results[:3]}")

    if local_wins:
        print(f"\n{'LOCAL WINS':} ({len(local_wins)} tasks - local gets right, semantic misses):")
        for r in local_wins:
            lr = local_by_id[r.task_id]
            print(f"  - {r.task_id}: '{r.query}'")
            print(f"    Local got:    {lr.top_results[:3]}")
            print(f"    Semantic got: {r.top_results[:3]}")

    print(f"\n{'BOTH MISS':} ({len(both_miss)} tasks):")
    for r in both_miss:
        sr = semantic_by_id[r.task_id]
        print(f"  - {r.task_id}: '{r.query}'")
        print(f"    Local got:    {r.top_results[:3]}")
        print(f"    Semantic got: {sr.top_results[:3]}")


def run_benchmark(
    api_key: str | None = None,
    base_url: str = "https://api.stackone.com",
    use_local: bool = False,
    local_lambda_url: str = DEFAULT_LOCAL_LAMBDA_URL,
) -> ComparisonReport:
    """Run the full benchmark comparison.

    Args:
        api_key: StackOne API key (uses STACKONE_API_KEY env var if not provided)
        base_url: Base URL for production API requests
        use_local: If True, use local Lambda instead of production API
        local_lambda_url: URL of local Lambda endpoint

    Returns:
        ComparisonReport with results

    Raises:
        ValueError: If no API key is available (production mode only)
    """
    # Create semantic search client and load tools based on mode
    if use_local:
        print(f"Using LOCAL Lambda at: {local_lambda_url}")
        local_client = LocalLambdaSearchClient(lambda_url=local_lambda_url)
        semantic_client: SearchClientProtocol = local_client

        # Fetch tool catalog from the Lambda itself (no /mcp or API key needed)
        print("Fetching action catalog from local Lambda...")
        actions = local_client.fetch_all_actions()
        tools = [LightweightTool(name=a.action_name, description=a.description) for a in actions]
        print(f"Loaded {len(tools)} actions from Lambda")
    else:
        api_key = api_key or os.environ.get("STACKONE_API_KEY")
        if not api_key:
            raise ValueError("API key must be provided or set via STACKONE_API_KEY environment variable")
        print(f"Using PRODUCTION API at: {base_url}")
        semantic_client = SemanticSearchClient(api_key=api_key, base_url=base_url)

        from stackone_ai import StackOneToolSet

        print("Initializing toolset...")
        toolset = StackOneToolSet(api_key=api_key, base_url=base_url)

        print("Fetching tools (this may take a moment)...")
        tools = list(toolset.fetch_tools())
        print(f"Loaded {len(tools)} tools")

    print(f"\nRunning benchmark with {len(EVALUATION_TASKS)} evaluation tasks...")
    benchmark = SearchBenchmark(tools, semantic_client=semantic_client)

    report = benchmark.compare()
    print_report(report)

    return report


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Benchmark comparing local BM25+TF-IDF vs semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with production API
  STACKONE_API_KEY=xxx python tests/benchmark_search.py

  # Run with local Lambda (start it first: cd ai-generation/apps/action_search && make run-local)
  python tests/benchmark_search.py --local

  # Run with custom local Lambda URL
  python tests/benchmark_search.py --local --lambda-url http://localhost:9000/invoke
        """,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local Lambda instead of production API",
    )
    parser.add_argument(
        "--lambda-url",
        default=DEFAULT_LOCAL_LAMBDA_URL,
        help=f"Local Lambda URL (default: {DEFAULT_LOCAL_LAMBDA_URL})",
    )
    parser.add_argument(
        "--api-url",
        default="https://api.stackone.com",
        help="Production API base URL",
    )

    args = parser.parse_args()

    try:
        run_benchmark(
            base_url=args.api_url,
            use_local=args.local,
            local_lambda_url=args.lambda_url,
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Set STACKONE_API_KEY environment variable or use --local flag")
        exit(1)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
