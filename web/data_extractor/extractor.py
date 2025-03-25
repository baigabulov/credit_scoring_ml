from typing import Optional, Any

from openai import OpenAI
from pydantic import BaseModel

from django.conf import settings


class ScoringData(BaseModel):
    # 1. Описание бизнеса
    industry_and_market: Optional[str]
    unique_selling_proposition: Optional[str]
    business_model: Optional[str]

    # 2. Финансовые данные (прогнозные)
    expected_revenue: Optional[float]
    expected_profit: Optional[float]
    cost_structure: Optional[dict[str, float]]
    funding_plan: Optional[dict[str, float]]

    # 3. Риски и стратегия их минимизации
    competitors: Optional[list[str]]
    growth_barriers: Optional[list[str]]
    risk_mitigation_strategy: Optional[str]

    # 4. Команда и опыт
    founders_experience: Optional[list[float]]
    key_employees: Optional[list[str]]
    previous_projects: Optional[list[str]]

    # 5. Требуемая сумма кредита и цели использования
    requested_loan_amount: Optional[float]
    loan_purpose: Optional[str]
    expected_investment_effect: Optional[str]

    # 6. План возврата кредита
    repayment_sources: Optional[list[str]]
    repayment_schedule: Optional[dict[str, Any]]
    collateral_or_guarantees: Optional[str]


class DataExtractor:

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            project=settings.OPENAI_PROJECT_ID,
        )

    @property
    def prompt_instruction(self):
        variables = ','.join([
            'industry_and_marketunique_selling_proposition', 'business_model', 'expected_revenue', 'expected_profit',
            'cost_structure', 'funding_plan', 'competitors', 'growth_barriers', 'risk_mitigation_strategy', 'founders_experience',
            'key_employees', 'previous_projects', 'requested_loan_amount', 'loan_purpose', 'expected_investment_effect',
            'repayment_sources', 'repayment_schedule', 'collateral_or_guarantees',
        ])
        return (
            'Тебе нужно достать из текста все переменные. '
            f'Переменные: {variables}'
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
