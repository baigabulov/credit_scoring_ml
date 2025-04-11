import random

from ui.models import ScoringRequest


class Scorer:
    def __init__(self, scoring_request: ScoringRequest):
        self.scoring_request = scoring_request

    def score(self) -> str:
        if self.scoring_request.status == ScoringRequest.STATUS_DONE:
            return ScoringRequest.STATUS_DONE
        
        state = self.scoring_request.request_data.get('state', None)
        term = random.randint(1, 10) * 4
        approved_amount = random.randint(0, int(self.scoring_request.request_data['loan_amount']))
        
        percent = random.randint(1, 16)
        monthly_payment = round(approved_amount * (1 + percent / 100) / term, 2)
        is_approved = round(approved_amount / float(self.scoring_request.request_data['loan_amount']), 2) > 0.4

        if not state:
            self.scoring_request.request_data['state'] = 1
            self.scoring_request.request_data[ScoringRequest.MODEL_PRL] = {
                'term': term,
                'approved_amount': approved_amount,
                'monthly_payment': monthly_payment,
                'percent': percent,
                'overpayment': round(monthly_payment * term - approved_amount, 2),
                'result': 'approved' if is_approved else 'rejected',
            }
            self.scoring_request.save(update_fields=['request_data'])
            return ScoringRequest.STATUS_PROCESSING
        
        if state == 1:
            self.scoring_request.request_data['state'] = 2
            self.scoring_request.request_data[ScoringRequest.MODEL_PSML] = {
                'term': term,
                'approved_amount': approved_amount,
                'monthly_payment': monthly_payment,
                'percent': percent,
                'overpayment': round(monthly_payment * term - approved_amount, 2),
                'result': 'approved' if is_approved else 'rejected',
            }
            self.scoring_request.save(update_fields=['request_data'])
            return ScoringRequest.STATUS_PROCESSING
        
        if state == 2:
            self.scoring_request.request_data['state'] = 3
            self.scoring_request.request_data[ScoringRequest.MODEL_EMPCC] = {
                'term': term,
                'approved_amount': approved_amount,
                'monthly_payment': monthly_payment,
                'percent': percent,
                'overpayment': round(monthly_payment * term - approved_amount, 2),
                'result': 'approved' if is_approved else 'rejected',
            }
            self.scoring_request.save(update_fields=['request_data'])
            return ScoringRequest.STATUS_PROCESSING
        
        chosen_model = None
        chosen_model_approved_amount = 0
        for model in [ScoringRequest.MODEL_PRL, ScoringRequest.MODEL_PSML, ScoringRequest.MODEL_EMPCC]:
            if self.scoring_request.request_data[model]['result'] == 'approved':
                if self.scoring_request.request_data[model]['approved_amount'] > chosen_model_approved_amount:
                    chosen_model = model
                    chosen_model_approved_amount = self.scoring_request.request_data[model]['approved_amount']

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
