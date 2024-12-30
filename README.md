# Introduction

Joinly is a library designed join two different lists of keyed values using a large language model (LLM). Like a human data annotator, you can provide context through a prompt to specify how you want the task accomplished. The library supports inner, left, right, and full joins.

By default, the library uses OpenAI's GPT-4o model. To use the library, set your OpenAI credentials as follows:

```
OPENAI_API_KEY=XXXX
```

# Format

The functions expect two keyed lists. These are lists of tuples where the first element of each tuple is a string key, and the second element is a value returned but not directly used in the join. The results are returned as a list of paired tuples for the left and right sides.

The types are explicitly defined in Python as:

```
from joinly import join

left: List[Tuple[str, Any]] = []
right: List[Tuple[str, Any]] = []

results: List[Tuple[Tuple[str, Any], Tuple[str, Any]]] = join.inner_join(left, right)
```

# Installation

Install Joinly using pip:

```
pip install joinly
```

# Quickstart

Suppose you have two lists of fruits. One is more specific about the types of fruits, while the other is more general. Joinly allows you to describe the task and join the lists in a fuzzy manner using LLMs.

```
from joinly import join

context = '''
You are joining fruits, so make sure to account for different types of fruits, ensuring the categories match correctly.
'''

left = [
    ("apple", 1),
    ("banana", 2),
    ("orange", 3),
]

right = [
    ("honeycrisp", 4),
    ("florida orange", 5),
    ("gala", 6),
    ("grannysmith", 7)
]

results = join.inner_join(left, right, context=context)
> [
    (("apple", 1), ("honeycrisp", 4)),
    (("apple", 1), ("gala", 6)),
    (("apple", 1), ("grannysmith", 7)),
    (("orange", 3), ("florida orange", 5))
]
```

# Custom LLMs

You can integrate any custom model by implementing the `BaseAI` class with an embedding method and a prompt.

```
from joinly.ai import BaseAI

class MyAI(BaseAI):
    def __init__(self) -> None:
        self.client = ...

    def __call__(self, system_message: str, user_message: str) -> Optional[str]:
        pass

    def embed(self, text: str) -> Optional[ArrayLike]:
        pass

results = join.inner_join(left, right, context=context, llm=MyAI())
```
