import unittest
from langchain_core.messages import HumanMessage
from agent import build_agent

#colors

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    GREEN   = "\033[92m"
    RED     = "\033[91m"
    CYAN    = "\033[96m"
    YELLOW  = "\033[93m"
    GREY    = "\033[90m"
    WHITE   = "\033[97m"


#helpers

def run_agent(question: str) -> dict:
    agent = build_agent()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})

    messages = result["messages"]
    answer = messages[-1].content if messages else ""

    tool_calls = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(tc["name"])

    return {"answer": answer, "tool_calls": tool_calls, "messages": messages}


def print_test_block(tool: str, question: str, answer: str, passed: bool, fail_reason: str = ""):
    """Prints a formatted, colored block for a single test result."""
    status_label = f"{C.GREEN}{C.BOLD}  PASS  {C.RESET}" if passed else f"{C.RED}{C.BOLD}  FAIL  {C.RESET}"
    border = C.GREEN + "─" * 70 + C.RESET if passed else C.RED + "─" * 70 + C.RESET

    print(border)
    print(f" {status_label}  {C.BOLD}{C.CYAN}Tool:{C.RESET} {C.WHITE}{tool}{C.RESET}")
    print(f"\n {C.YELLOW}Question:{C.RESET}")
    print(f"   {question}")
    print(f"\n {C.YELLOW}Agent Answer:{C.RESET}")
    for line in answer.strip().splitlines():
        print(f"   {C.GREY}{line}{C.RESET}")
    if not passed and fail_reason:
        print(f"\n {C.RED}Failure Reason:{C.RESET}")
        print(f"   {C.RED}{fail_reason}{C.RESET}")
    print(border)
    print()


class PrintingTestCase(unittest.TestCase):
    """Base class that wraps each test so it prints a formatted block regardless of pass or fail."""

    def _run(self, tool: str, question: str, assertions_fn):
        result = run_agent(question)
        try:
            assertions_fn(result)
            print_test_block(tool, question, result["answer"], passed=True)
        except AssertionError as e:
            print_test_block(tool, question, result["answer"], passed=False, fail_reason=str(e))
            raise


# tests
class TestSemanticSearch(PrintingTestCase):

    def test_find_python_backend_jobs(self):
        def check(r):
            self.assertIn("semantic_search_jobs", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="semantic_search_jobs",
            question="Find me Python backend developer jobs",
            assertions_fn=check,
        )

    def test_find_ml_engineer_jobs(self):
        def check(r):
            self.assertIn("semantic_search_jobs", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="semantic_search_jobs",
            question="I'm a machine learning engineer with 3 years of experience. What roles match my profile?",
            assertions_fn=check,
        )

    def test_find_fintech_jobs(self):
        def check(r):
            self.assertIn("semantic_search_jobs", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="semantic_search_jobs",
            question="Show me jobs at fintech startups",
            assertions_fn=check,
        )


class TestGetJobDetails(PrintingTestCase):

    def test_detail_request_after_search(self):
        def check(r):
            called = r["tool_calls"]
            self.assertTrue(
                "semantic_search_jobs" in called or "search_jobs_by_criteria" in called,
                f"Expected a search tool. Got: {called}",
            )
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="get_job_details",
            question="Search for frontend React jobs and then show me the full details of the first result you find.",
            assertions_fn=check,
        )


class TestGetJobAggregate(PrintingTestCase):

    def test_average_experience_data_scientists(self):
        def check(r):
            self.assertIn("get_job_aggregate", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="get_job_aggregate (AVG)",
            question="What is the average years of experience required for Data Scientist jobs?",
            assertions_fn=check,
        )

    def test_count_all_jobs(self):
        def check(r):
            self.assertIn("get_job_aggregate", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="get_job_aggregate (COUNT)",
            question="How many job postings are in the database?",
            assertions_fn=check,
        )

    def test_min_experience_backend(self):
        def check(r):
            self.assertIn("get_job_aggregate", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="get_job_aggregate (MIN)",
            question="What is the minimum experience required for backend developer roles?",
            assertions_fn=check,
        )

    def test_max_experience_required(self):
        def check(r):
            self.assertIn("get_job_aggregate", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="get_job_aggregate (MAX)",
            question="What is the maximum years of experience any job in the database requires?",
            assertions_fn=check,
        )

    def test_column_mapping_experience_synonym(self):
        def check(r):
            self.assertIn("get_job_aggregate", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="get_job_aggregate (synonym mapping: 'tenure')",
            question="What is the average background tenure needed for a DevOps engineer?",
            assertions_fn=check,
        )


class TestGetColumnDistribution(PrintingTestCase):

    def test_top_locations(self):
        def check(r):
            self.assertIn("get_column_distribution", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="get_column_distribution (location)",
            question="What are the top job locations in the database?",
            assertions_fn=check,
        )

    def test_top_companies(self):
        def check(r):
            self.assertIn("get_column_distribution", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="get_column_distribution (company)",
            question="Which companies have the most job postings?",
            assertions_fn=check,
        )

    def test_seniority_breakdown(self):
        def check(r):
            self.assertIn("get_column_distribution", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="get_column_distribution (seniority)",
            question="Give me a breakdown of jobs by seniority level.",
            assertions_fn=check,
        )

    def test_role_distribution(self):
        def check(r):
            self.assertIn("get_column_distribution", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="get_column_distribution (role)",
            question="What are the most common job roles in the database?",
            assertions_fn=check,
        )

    def test_experience_distribution(self):
        def check(r):
            self.assertIn("get_column_distribution", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="get_column_distribution (yearsexperience)",
            question="Show me the distribution of years of experience across all jobs.",
            assertions_fn=check,
        )


class TestSearchJobsByCriteria(PrintingTestCase):

    def test_filter_by_location(self):
        def check(r):
            self.assertIn("search_jobs_by_criteria", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="search_jobs_by_criteria (location)",
            question="Show me software engineering jobs in Tel Aviv.",
            assertions_fn=check,
        )

    def test_filter_by_company(self):
        def check(r):
            self.assertIn("search_jobs_by_criteria", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="search_jobs_by_criteria (company)",
            question="Are there any jobs at Google in the database?",
            assertions_fn=check,
        )

    def test_filter_by_max_experience(self):
        def check(r):
            self.assertIn("search_jobs_by_criteria", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="search_jobs_by_criteria (max_experience)",
            question="I'm a junior developer. Find me jobs that require at most 2 years of experience.",
            assertions_fn=check,
        )

    def test_filter_by_role_and_location(self):
        def check(r):
            self.assertIn("search_jobs_by_criteria", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="search_jobs_by_criteria (role + location)",
            question="Find React developer jobs in New York.",
            assertions_fn=check,
        )


class TestTopSkills(PrintingTestCase):

    def test_top_skills_for_role(self):
        def check(r):
            self.assertIn("top_skills", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="top_skills (specific role)",
            question="What are the top skills I need to become a DevOps engineer?",
            assertions_fn=check,
        )

    def test_top_skills_for_data_science(self):
        def check(r):
            self.assertIn("top_skills", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="top_skills (data science)",
            question="What technical skills are most in-demand for Data Science roles?",
            assertions_fn=check,
        )

    def test_top_skills_all_roles(self):
        def check(r):
            self.assertIn("top_skills_all", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="top_skills_all",
            question="What are the top skills across all job listings in the database?",
            assertions_fn=check,
        )

    def test_top_skills_overall_market(self):
        def check(r):
            self.assertIn("top_skills_all", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="top_skills_all (market competitiveness)",
            question="If I want to be competitive in the tech job market, what skills should I focus on?",
            assertions_fn=check,
        )


class TestEdgeCases(PrintingTestCase):

    def test_ambiguous_experience_wording(self):
        def check(r):
            self.assertIn("get_job_aggregate", r["tool_calls"], f"Tool not called. Got: {r['tool_calls']}")
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="get_job_aggregate (synonym: 'tenure')",
            question="On average, how many years of tenure do companies expect from a software developer?",
            assertions_fn=check,
        )

    def test_broad_keyword_role_filter(self):
        def check(r):
            called = r["tool_calls"]
            self.assertTrue(
                "get_job_aggregate" in called or "search_jobs_by_criteria" in called,
                f"Expected aggregate or criteria tool. Got: {called}",
            )
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="get_job_aggregate / search_jobs_by_criteria (multi-word role)",
            question="What is the average experience required for a full stack developer?",
            assertions_fn=check,
        )

    def test_no_fabricated_data(self):
        def check(r):
            answer = r["answer"].lower()
            self.assertTrue(
                any(p in answer for p in [
                    "no ", "not found", "couldn't find", "unable", "don't have",
                    "general industry knowledge", "0 result", "no jobs",
                ]),
                "Agent did not admit missing data — possible hallucination.",
            )
        self._run(
            tool="(no tool — fallback expected)",
            question="List every job at AcmeXYZCorp999 that requires 47 years of experience.",
            assertions_fn=check,
        )

    def test_answer_always_returned(self):
        def check(r):
            self.assertTrue(r["answer"].strip(), "Empty answer")
        self._run(
            tool="(any / none)",
            question="Tell me something interesting about the job market.",
            assertions_fn=check,
        )



if __name__ == "__main__":
    unittest.main(verbosity=0)