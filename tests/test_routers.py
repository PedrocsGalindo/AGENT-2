import unittest

from academic_explorer_mvp.graph.routers import route_after_decision


class RouterTests(unittest.TestCase):
    def test_route_after_decision_finalizes_when_stop_reason_exists(self) -> None:
        self.assertEqual(route_after_decision({"stop_reason": "max rounds reached"}), "finalize")

    def test_route_after_decision_continues_without_stop_reason(self) -> None:
        self.assertEqual(route_after_decision({"stop_reason": None}), "continue")


if __name__ == "__main__":
    unittest.main()
