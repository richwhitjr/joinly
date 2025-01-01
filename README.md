# Introduction

Joinly is a library designed join two different lists of keyed values using a large language model (LLM). Like a human data annotator, you can provide context through a prompt to specify how you want the task accomplished. The library supports inner, left, right, and full joins.

# Installation

Install Joinly using pip:

```
pip install joinly
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

By default, the library uses OpenAI's GPT-4o model. To use the library, set your OpenAI credentials as follows:

```
OPENAI_API_KEY=XXXX
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

# Cost and Complexity

This library leverages LLMs to perform fuzzy matching on keys. In the worst case, the complexity is O(M * N), where M is the length of one list and N is the length of the second list. This represents the most computationally expensive scenario.

To mitigate complexity and reduce costs, the library uses text embeddings as a pre-filter before invoking the LLM for direct comparisons. Embeddings are computed for each item once and cached, reducing complexity to O(M + N). The cosine similarity of the embeddings is then used as a pre-filter. By default, the library calls the LLM on keyed pairs with a similarity greater than 0.6.

If you expect tighter embedding similarities, you can increase this threshold using the `embedding_threshold` argument:

```
from joinly import join

left = []
right = []

join.inner_join(left, right, embedding_threshold=0.9)
```

# Validation

Often, results can be improved by performing a second pass to check the outcomes of the first pass. Joinly has a built-in function called validate that does exactly this for you. It takes the results of any joins and, when it finds a matching pair, passes it through the validator function, asking the LLM to check the answer. The resulting list provides both a True/False indicator for a match and a reason for the result.

```
from joinly import join

left = []
right = []

results = join.inner_join(left, right)

validation = join.validate(results)
```
