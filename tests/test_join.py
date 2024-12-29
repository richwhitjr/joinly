from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from src.common.llm import join


class FakeAI(join.BaseAI):
    def __init__(self, response="FALSE"):
        self.response = response

    def __call__(self, system_message, user_message) -> Optional[str]:
        return self.response

    def embed(self, text: str) -> Optional[ArrayLike]:
        return np.array([1.0, 0.0, 0.0])


def test_join():
    context = "You are joining fruits so make sure to be aware of that fruits have different types but make sure the catagory is correct"

    left = [("apple", 1), ("banana", 2), ("orange", 3)]

    right = [("honeycrisp", 4), ("florida orange", 5), ("gala", 6), ("grannysmith", 7)]

    results = join.inner_join(left, right, context=context, llm=FakeAI(response="FALSE"))

    assert len(results) == 0

    results = join.inner_join(left, right, context=context, llm=FakeAI(response="TRUE"))

    assert len(results) == 12

    results = join.left_join(left, right, context=context, llm=FakeAI(response="FALSE"))

    assert len(results) == 3

    results = join.right_join(left, right, context=context, llm=FakeAI(response="FALSE"))

    assert len(results) == 4

    results = join.full_join(left, right, context=context, llm=FakeAI(response="FALSE"))

    assert len(results) == 7
