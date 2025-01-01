"""Join two lists of items using a language model."""

# ruff: noqa: SIM103

import dataclasses
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np
import tqdm
from numpy.typing import ArrayLike

from joinly import prompts
from joinly.ai import BaseAI, OpenAI

logger = logging.getLogger(__name__)


ITEM_TYPE = tuple[str, Any]
EMBED_TIME_TYPE = tuple[tuple[str, ArrayLike], Any]
VALUE_TYPE = list[ITEM_TYPE]
LIST_EMBED_TYPE = list[EMBED_TIME_TYPE]
COGROUP_VALUE_TYPE = tuple[VALUE_TYPE, VALUE_TYPE]


def process_embedding(item: tuple[str, Any], llm: BaseAI) -> tuple[tuple[str, Optional[ArrayLike]], Any]:
    text, label = item
    vec = llm.embed(text)
    return ((text, vec), label)


def embed(
    items: list[tuple[str, Any]], llm: Optional[BaseAI] = None
) -> list[tuple[tuple[str, Optional[ArrayLike]], Any]]:
    if llm is None:
        llm = OpenAI()
    process_func = functools.partial(process_embedding, llm=llm)
    embeddings = []
    with ThreadPoolExecutor() as executor:
        futures = list(
            tqdm.tqdm(
                executor.map(process_func, items),
                total=len(items),
                desc="Embedding items",
            )
        )
        embeddings.extend(futures)
    return embeddings


def _cosine(v1: ArrayLike, v2: ArrayLike) -> Any:
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def _process_embedding_pair(pair: EMBED_TIME_TYPE, context: str, llm: BaseAI) -> Optional[tuple[ITEM_TYPE, ITEM_TYPE]]:
    (lk, lv), (rk, rv) = pair
    match = matcher((lk, lv), (rk, rv), context, llm)
    if match:
        return (lk, lv), (rk, rv)
    return None


def _parallel_process_embed(
    left_list: list[ITEM_TYPE],
    right_list: list[ITEM_TYPE],
    context: str,
    llm: BaseAI,
    embedding_treshold: float,
) -> list[tuple[ITEM_TYPE, ITEM_TYPE]]:
    process_func = functools.partial(_process_embedding_pair, context=context, llm=llm)
    results = []
    right_embedding = embed(right_list, llm)
    with ThreadPoolExecutor() as executor:
        futures = []
        for left in tqdm.tqdm(left_list, leave=False):
            le = llm.embed(left[0])
            for right in right_embedding:
                ((rk, re), rv) = right
                if re is None or le is None:
                    continue
                dist = _cosine(le, re)
                if dist < embedding_treshold:
                    continue
                future = executor.submit(process_func, (left, (rk, rv)))
                futures.append(future)
        for future in tqdm.tqdm(futures):
            result = future.result()
            if result:
                left_v, right_v = result
                results.append((left_v, right_v))
    return results


def inner_join(
    left: list[ITEM_TYPE],
    right: list[ITEM_TYPE],
    context: str = "",
    llm: Optional[BaseAI] = None,
    embedding_treshold: float = 0.6,
) -> list[tuple[ITEM_TYPE, ITEM_TYPE]]:
    """Inner join two lists of items using a language model.
    Args:
        left: list of items to join.
        right: list of items to join.
        context: Context for the join.
        llm: Language model to use for the join.
        embedding_treshold: Minimum distance between embeddings for a match.
    Returns:
        list of tuples of matched items.
    """
    if llm is None:
        llm = OpenAI()
    matches = _parallel_process_embed(left, right, context, llm, embedding_treshold)
    return matches


def left_join(
    left: list[ITEM_TYPE],
    right: list[ITEM_TYPE],
    context: str = "",
    llm: Optional[BaseAI] = None,
    embedding_treshold: float = 0.6,
) -> list[tuple[ITEM_TYPE, Optional[ITEM_TYPE]]]:
    """Left join two lists of items using a language model.
    Args:
        left: list of items to join.
        right: list of items to join.
        context: Context for the join.
        llm: Language model to use for the join.
        embedding_treshold: Minimum distance between embeddings for a match.
    Returns:
        list of tuples of matched items.
    """
    inner = inner_join(left, right, context, llm, embedding_treshold)
    return_items: list[tuple[ITEM_TYPE, Optional[ITEM_TYPE]]] = []
    for left_row in left:
        found = False
        for i in inner:
            if left_row == i[0]:
                return_items.append(i)
                found = True
                break
        if not found:
            return_items.append((left_row, None))
    return return_items + list(inner)


def right_join(
    left: list[ITEM_TYPE],
    right: list[ITEM_TYPE],
    context: str = "",
    llm: Optional[BaseAI] = None,
    embedding_treshold: float = 0.6,
) -> list[tuple[Optional[ITEM_TYPE], ITEM_TYPE]]:
    """Right join two lists of items using a language model.
    Args:
        left: list of items to join.
        right: list of items to join.
        context: Context for the join.
        llm: Language model to use for the join.
        embedding_treshold: Minimum distance between embeddings for a match.
    Returns:
        list of tuples of matched items.
    """
    inner = inner_join(left, right, context, llm, embedding_treshold)
    return_items: list[tuple[Optional[ITEM_TYPE], ITEM_TYPE]] = []
    for right_row in right:
        found = False
        for i in inner:
            if right_row == i[1]:
                return_items.append(i)
                found = True
                break
        if not found:
            return_items.append((None, right_row))
    return return_items + list(inner)


def full_join(
    left: list[ITEM_TYPE],
    right: list[ITEM_TYPE],
    context: str = "",
    llm: Optional[BaseAI] = None,
    embedding_treshold: float = 0.6,
) -> list[tuple[Optional[ITEM_TYPE], Optional[ITEM_TYPE]]]:
    """Full join two lists of items using a language model.
    Args:
        left: list of items to join.
        right: list of items to join.
        context: Context for the join.
        llm: Language model to use for the join.
        embedding_treshold: The minimum distance between embeddings for a match.
    Returns:
        list of tuples of matched items.
    """
    inner = inner_join(left, right, context, llm, embedding_treshold)
    return_items: list[tuple[Optional[ITEM_TYPE], Optional[ITEM_TYPE]]] = []
    for left_row in left:
        found = False
        for i in inner:
            if left_row == i[0]:
                return_items.append(i)
                found = True
                break
        if not found:
            return_items.append((left_row, None))
    for right_row in right:
        found = False
        for i in inner:
            if right_row == i[1]:
                found = True
                break
        if not found:
            return_items.append((None, right_row))
    return return_items + list(inner)


@dataclasses.dataclass
class ValidationItem:
    left: Optional[ITEM_TYPE]
    right: Optional[ITEM_TYPE]
    match: bool = True
    reason: Optional[str] = None


def _process_validate(pair: tuple[ITEM_TYPE, ITEM_TYPE], context: str, llm: BaseAI) -> ValidationItem:
    (lk, lv), (rk, rv) = pair
    response, answer = validator((lk, lv), (rk, rv), context, llm)
    return ValidationItem((lk, lv), (rk, rv), answer, response)


def validate(
    items: list[tuple[Optional[ITEM_TYPE], Optional[ITEM_TYPE]]],
    context: str = "",
    llm: Optional[BaseAI] = None,
) -> list[ValidationItem]:
    """Validate a list of items using a language model. The validation checks the response of a matcher.
    Args:
        items: List of items to validate.
        context: Context for the validation.
        llm: Language model to use for the validation.
    Returns:
        List of validation items.
    """
    if llm is None:
        llm = OpenAI()
    process_func = functools.partial(_process_validate, context=context, llm=llm)
    with ThreadPoolExecutor() as executor:
        futures = []
        return_items = []
        for left, right in tqdm.tqdm(items):
            if left is None or right is None:
                return_items.append(ValidationItem(left, right))
                continue
            if left is None:
                return_items.append(ValidationItem(None, right))
                continue
            if right is None:
                return_items.append(ValidationItem(left, None))
                continue
            future = executor.submit(process_func, (left, right))
            futures.append(future)

        for future in tqdm.tqdm(futures):
            result = future.result()
            return_items.append(result)

        return return_items


def validator(
    left: tuple[str, Any],
    right: tuple[str, Any],
    context: str = "",
    llm: Optional[BaseAI] = None,
) -> tuple[Optional[str], bool]:
    """Match two items using a language model.
    Args:
        left: Item to match.
        righat: Item to match.
        context: Context for the match.
        llm: Language model to use for the match.
    Returns:
        True if the items match, False otherwise.
    """
    if llm is None:
        llm = OpenAI()
    prompt = prompts.DEFAULT_VALIDATOR_PROMPT.format(context=context)
    answer = llm(prompt, f"left = {left[0]!s}, right={right[0]!s}")
    if answer is None:
        return (answer, False)
    flag = llm(prompts.DEFAULT_VALIDATOR_PROMPT_CHECK_PROMPT, answer)
    if flag is None:
        return (answer, False)
    if "FALSE" in flag:
        return (answer, False)
    return (answer, True)


def matcher(
    left: tuple[str, Any],
    right: tuple[str, Any],
    context: str = "",
    llm: Optional[BaseAI] = None,
) -> bool:
    """Match two items using a language model.
    Args:
        left: Item to match.
        right: Item to match.
        context: Context for the match.
        llm: Language model to use for the match.
    Returns:
        True if the items match, False otherwise.
    """
    if llm is None:
        llm = OpenAI()
    prompt = prompts.DEFAULT_JOIN_PROMPT.format(context=context)
    answer = llm(prompt, f"left = {left[0]!s}, right={right[0]!s}")
    logger.debug(answer)
    if answer is None:
        return False
    if "FALSE" in answer:
        return False
    return True


def debug(
    left: tuple[str, Any],
    right: tuple[str, Any],
    context: str = "",
    llm: Optional[BaseAI] = None,
) -> Optional[str]:
    """Debug two items using a language model.
    Args:
        left: Item to match.
        right: Item to match.
        context: Context for the match.
        llm: Language model to use for the match.
    Returns:
        The reason the items match or do not match.
    """
    if llm is None:
        llm = OpenAI()
    prompt = prompts.DEFAULT_DEBUG_PROMPT.format(context=context)
    answer = llm(prompt, f"left = {left[0]!s}, right={right[0]!s}")
    return answer
