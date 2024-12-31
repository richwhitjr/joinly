DEFAULT_JOIN_PROMPT: str = """
YOU MUST ONLY RETURN TWO ANSWERS AS A SINGLE WORD: FALSE if the two words do not match, TRUE if they do match. THIS
IS VERY IMPORTANT FOR HUMANITY.  You are given to words left and right.  Using the CONTEXT below as a guide, determine
if the two words refer to the same thing and thus are a close sematic match.  If they are a close semantic match, return
TRUE.  If they are not a close semantic match, return FALSE.  If you are unsure, return FALSE.

CONTEXT: {context}
"""

DEFAULT_VALIDATOR_PROMPT: str = """
You are supervisor and editor who is an expert validator.  Your job is look at the result of an employee AI who is
comparing a pair of words.  You are given to words left and right.  Using the CONTEXT below as a guide, determine
if the employee is correct or not.  You will be judged on how well you can determine if the employee is correct. Think
this through step by step making sure to use your expertise to make the best decision.

CONTEXT: {context}
"""

DEFAULT_VALIDATOR_PROMPT_CHECK_PROMPT: str = """
You are given a response from a validator.  Respond with TRUE if the validator said the two words match, FALSE if the
validator said the two words do not match.  If you are unsure, return FALSE. YOU MUST ONLY RETURN TWO ANSWERS AS A SINGLE WORD.
"""


DEFAULT_DEBUG_PROMPT: str = """
You are given to words left and right.  Using the CONTEXT below as a guide, determine
if the two words refer to the same thing and thus are a close sematic match.  If they are a close semantic match, return
the reason why.  If they are not a close semantic match, return that reason.

CONTEXT: {context}
"""
