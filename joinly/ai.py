import logging
from typing import Optional

import numpy as np
import openai
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


class BaseAI:
    def __call__(self, system_message: str, user_message: str) -> Optional[str]:
        return None

    def embed(self, text: str) -> Optional[ArrayLike]:
        return None


class OpenAI(BaseAI):
    def __init__(self) -> None:
        self.client = openai.OpenAI()

    def __call__(self, system_message: str, user_message: str) -> Optional[str]:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_message}],
                },
            ],
        )
        prompt = response.choices[0].message.content
        logger.debug(prompt)
        return prompt

    def embed(self, text: str) -> Optional[ArrayLike]:
        return np.array(self.client.embeddings.create(input=[text], model="text-embedding-ada-002").data[0].embedding)
