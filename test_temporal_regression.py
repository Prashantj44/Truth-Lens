import os
import sys
import unittest
from unittest.mock import patch

os.environ["VERCEL"] = "1"
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.services.agents.agent_orchestrator import agent_orchestrator

class TestTemporalRegression(unittest.TestCase):
    def test_historical_fact_present_tense(self):
        # Present tense claim about a past fact
        claim = "Shinzo Abe is the Prime Minister of Japan."
        result = agent_orchestrator.process_claim(claim)
        verdict = result.get("verdict", "")
        self.assertIn(verdict, ["OUTDATED", "CONTRADICTED", "AMBIGUOUS_TIME_CONTEXT", "INSUFFICIENT_EVIDENCE", "SUPPORTED_HISTORICALLY"])
        self.assertNotEqual(verdict, "SUPPORTED_CURRENT")

    def test_historical_fact_past_tense(self):
        # Past tense claim about a past fact should be supported
        claim = "Shinzo Abe was the Prime Minister of Japan."
        result = agent_orchestrator.process_claim(claim)
        verdict = result.get("verdict", "")
        self.assertIn(verdict, ["SUPPORTED_CURRENT", "SUPPORTED_HISTORICALLY", "AMBIGUOUS_TIME_CONTEXT"])

    def test_demo_overrides_respects_time(self):
        claim = "The neighbouring galaxy is called earth"
        result = agent_orchestrator.process_claim(claim)
        verdict = result.get("verdict", "")
        self.assertEqual(verdict, "CONTRADICTED")

if __name__ == "__main__":
    unittest.main()
