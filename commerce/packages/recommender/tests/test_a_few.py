import unittest

from commerce.evaluation.a_few import split


def visit(customer, rank, items):
    return {"customer_id_local": customer, "order_rank": rank, "items": [{"item_id_local": i} for i in items]}


class Split(unittest.TestCase):
    def test_each_last_visit_is_the_answer_and_never_in_the_ledger(self):
        events = [visit("c1", 2, ["b"]), visit("c1", 1, ["a"]), visit("c1", 3, ["c", "d"]),
                  visit("c2", 1, ["a"])]
        ledger, answers = split(events)
        self.assertEqual(answers, {"c1": {"c", "d"}})
        # c2 has one visit: no answer, but its visit is part of the seller's past.
        self.assertEqual(sorted((e["customer_id_local"], e["order_rank"]) for e in ledger),
                         [("c1", 1), ("c1", 2), ("c2", 1)])


if __name__ == "__main__":
    unittest.main()
