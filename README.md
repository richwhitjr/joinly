

# Introduction

Joinly is a library that is mean to help assist in the joining of two different lists of items using a LLM. Like a
human data annotator you can provide context through a prompt on how you want the task achieved.  The library supports
inner, left, right, and full joins.

By default the library uses OpenAI's gpt-4o model.  You will need to set your OpenAI credentials prior to using the
library with:

```
 OPENAI_API_KEY=XXXX
```

# Installation

```
pip install joinly
```


# Quickstart

Let's say you have two lists of fruits.  One is a bit more specific with types of each fruit while the second
one is more general.  Joinly allows you to describe the task you want and join in a fuzzy way using LLMs.

```
from joinly import join

context = '''
You are joining fruits so make sure to be aware of that fruits
have different types but make sure the catagory is correct
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
 (('apple', 1), ('honeycrisp', 4)),
 (('apple', 1), ('gala', 6)),
 (('apple', 1), ('grannysmith', 7)),
 (('orange', 3), ('florida orange', 5))
]
```

# Other Models

You can bring in any custom model as BaseAI with an embedding method and a prompt.

```
from joinly.ai import BaseAI


class MyAI(BaseAI):
    def __init__(self) -> None:
        self.client = ...

    def __call__(self, system_message: str, user_message: str) -> Optional[str]
        pass

    def embed(self, text: str) -> Optional[ArrayLike]:
        pass


results = join.inner_join(left, right, context=context, llm=MyAI())
```
