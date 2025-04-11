from typing import Optional, Any

from openai import OpenAI
from pydantic import BaseModel

from django.conf import settings


class ScoringData(BaseModel):
    monthly_revenue: Optional[float]
    annual_revenue: Optional[float]


class DataExtractor:

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            project=settings.OPENAI_PROJECT_ID,
        )

    @property
    def prompt_instruction(self):
        return (
            """
            Определи сумму всех пополнений (входящие транзакции). Определи сумму всех расходов (исходящие транзакции).
            Вычисли средний месячный оборот (monthly_revenue) как сумму всех пополнений и всех расходов за 1 месяц.
            Вычисли общий годовой оборот (annual_revenue) как сумму всех пополнений и всех расходов за год. Если данных за год нет, то вычисли общий годовой оборот как сумму оборота за существующие месяцы.
            """
        )

    def extract_data(self, tokenized_data: list[int]):
        response = self.client.beta.chat.completions.parse(
            model=settings.CHATGPT_MODEL,
            messages=[
                {'role': 'developer', 'content': self.prompt_instruction},
                {'role': 'user', 'content': f'Текст: {tokenized_data}'},
            ],
            response_format=ScoringData,
        )

        return response.choices[0].message.parsed
