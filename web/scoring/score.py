import random
import json
from ui.models import ScoringRequest
from scoring.model_learning import DecisionTreeTraining, SVMTraining, LinearRegressionTraining


class Scorer:
    def __init__(self, scoring_request: ScoringRequest):
        self.scoring_request = scoring_request
        self.dt_model = DecisionTreeTraining()
        self.svm_model = SVMTraining()
        self.lr_model = LinearRegressionTraining()

    def prepare_data(self) -> dict:
        """
        Подготовка данных для моделей
        """
        response_data = json.loads(self.scoring_request.response_data)
        return {
            'activity_type': self.scoring_request.request_data['activity_type'],
            'loan_amount': float(self.scoring_request.request_data['loan_amount']),
            'loan_purpose': self.scoring_request.request_data['loan_purpose'],
            'revenue': float(response_data['monthly_revenue']),
            'revenue_days': 30,
        }

    def calculate_loan_terms(self, approved_amount: float) -> dict:
        """
        Расчет параметров кредита
        """
        term = random.randint(1, 10) * 4  # срок в месяцах
        percent = 12  # процентная ставка
        monthly_payment = round(approved_amount * (1 + percent / 100) / term, 2)
        overpayment = round(monthly_payment * term - approved_amount, 2)
        
        return {
            'term': term,
            'percent': percent,
            'monthly_payment': monthly_payment,
            'overpayment': overpayment
        }

    def score(self) -> str:
        if self.scoring_request.status == ScoringRequest.STATUS_DONE:
            return ScoringRequest.STATUS_DONE
        
        state = self.scoring_request.request_data.get('state', None)
        data = self.prepare_data()
        
        # Получаем предсказания от каждой модели
        if not state:
            self.scoring_request.request_data['state'] = 1
            approved_amount = self.dt_model.predict(data)
            terms = self.calculate_loan_terms(approved_amount)
            
            self.scoring_request.request_data[ScoringRequest.MODEL_PRL] = {
                'term': terms['term'],
                'approved_amount': approved_amount,
                'monthly_payment': terms['monthly_payment'],
                'percent': terms['percent'],
                'overpayment': terms['overpayment'],
                'result': 'approved' if approved_amount > 0 else 'rejected',
            }
            self.scoring_request.save(update_fields=['request_data'])
            return ScoringRequest.STATUS_PROCESSING
        
        if state == 1:
            self.scoring_request.request_data['state'] = 2
            approved_amount = self.svm_model.predict(data)
            terms = self.calculate_loan_terms(approved_amount)
            
            self.scoring_request.request_data[ScoringRequest.MODEL_PSML] = {
                'term': terms['term'],
                'approved_amount': approved_amount,
                'monthly_payment': terms['monthly_payment'],
                'percent': terms['percent'],
                'overpayment': terms['overpayment'],
                'result': 'approved' if approved_amount > 0 else 'rejected',
            }
            self.scoring_request.save(update_fields=['request_data'])
            return ScoringRequest.STATUS_PROCESSING
        
        if state == 2:
            self.scoring_request.request_data['state'] = 3
            approved_amount = self.lr_model.predict(data)
            terms = self.calculate_loan_terms(approved_amount)
            
            self.scoring_request.request_data[ScoringRequest.MODEL_EMPCC] = {
                'term': terms['term'],
                'approved_amount': approved_amount,
                'monthly_payment': terms['monthly_payment'],
                'percent': terms['percent'],
                'overpayment': terms['overpayment'],
                'result': 'approved' if approved_amount > 0 else 'rejected',
            }
            self.scoring_request.save(update_fields=['request_data'])
            return ScoringRequest.STATUS_PROCESSING
        
        # Выбираем лучшую модель
        chosen_model = ScoringRequest.MODEL_EMPCC
        chosen_model_approved_amount = self.scoring_request.request_data[chosen_model]['approved_amount']
        # for model in [ScoringRequest.MODEL_PRL, ScoringRequest.MODEL_PSML, ScoringRequest.MODEL_EMPCC]:
        #     if self.scoring_request.request_data[model]['result'] == 'approved':
        #         if self.scoring_request.request_data[model]['approved_amount'] > chosen_model_approved_amount:
        #             chosen_model = model
        #             chosen_model_approved_amount = self.scoring_request.request_data[model]['approved_amount']

        # Сохраняем итоговый результат
        self.scoring_request.request_data['result'] = {
            'model': chosen_model,
            'approved_amount': chosen_model_approved_amount,
            'term': self.scoring_request.request_data[chosen_model]['term'],
            'monthly_payment': self.scoring_request.request_data[chosen_model]['monthly_payment'],
            'percent': self.scoring_request.request_data[chosen_model]['percent'],
            'overpayment': self.scoring_request.request_data[chosen_model]['overpayment'],
        }
        self.scoring_request.status = ScoringRequest.STATUS_DONE
        self.scoring_request.save(update_fields=['status', 'request_data'])
        return ScoringRequest.STATUS_DONE
