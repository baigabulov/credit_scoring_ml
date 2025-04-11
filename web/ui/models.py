import json

from django.db import models


class ScoringRequest(models.Model):
    STATUS_PROCESSING = 'processing'
    STATUS_DONE = 'done'

    STATUS_CHOICES = [
        (STATUS_PROCESSING, 'В обработке'),
        (STATUS_DONE, 'Готово'),
    ]

    ACTIVITY_TYPE_RETAIL = 'retail'
    ACTIVITY_TYPE_MANUFACTURING = 'manufacturing'
    ACTIVITY_TYPE_SERVICES = 'services'
    ACTIVITY_TYPE_CONSTRUCTION = 'construction'
    ACTIVITY_TYPE_AGRICULTURE = 'agriculture'
    ACTIVITY_TYPE_OTHER = 'other'

    ACTIVITY_TYPE_CHOICES = [
        (ACTIVITY_TYPE_RETAIL, 'Розничная торговля'),
        (ACTIVITY_TYPE_MANUFACTURING, 'Производство'),
        (ACTIVITY_TYPE_SERVICES, 'Услуги'),
        (ACTIVITY_TYPE_CONSTRUCTION, 'Строительство'),
        (ACTIVITY_TYPE_AGRICULTURE, 'Сельское хозяйство'),
        (ACTIVITY_TYPE_OTHER, 'Другое'),
    ]

    LOAN_PURPOSE_WORKING_CAPITAL = 'working_capital'
    LOAN_PURPOSE_EQUIPMENT = 'equipment'
    LOAN_PURPOSE_EXPANSION = 'expansion'
    LOAN_PURPOSE_INVENTORY = 'inventory'
    LOAN_PURPOSE_OTHER = 'other'

    LOAN_PURPOSE_CHOICES = [
        (LOAN_PURPOSE_WORKING_CAPITAL, 'Пополнение оборотных средств'),
        (LOAN_PURPOSE_EQUIPMENT, 'Покупка оборудования'),
        (LOAN_PURPOSE_EXPANSION, 'Расширение бизнеса'),
        (LOAN_PURPOSE_INVENTORY, 'Покупка товаров'),
        (LOAN_PURPOSE_OTHER, 'Другое'),
    ]

    MODEL_PRL = 'DECISION_TREE'
    MODEL_PSML = 'SVM'
    MODEL_EMPCC = 'SNN'
    MODEL_CHOICES = [
        (MODEL_PRL, 'DECISION_TREE'),
        (MODEL_PSML, 'SMV'),
        (MODEL_EMPCC, 'SNN'),
    ]

    tokenized_data = models.TextField()
    request_data = models.JSONField(null=True, blank=True)
    response_data = models.JSONField(null=True, blank=True)
    is_completed = models.BooleanField(default=False)
    status = models.CharField(max_length=255, null=True, blank=True, choices=STATUS_CHOICES)
    created_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_date']
        verbose_name = 'Timestamp for request'
        verbose_name_plural = 'Timestamps for requests'

    def get_activity_type(self):
        if not self.request_data:
            return self.ACTIVITY_TYPE_OTHER
        return dict(self.ACTIVITY_TYPE_CHOICES).get(self.request_data.get('activity_type'), self.ACTIVITY_TYPE_OTHER)

    def get_loan_purpose(self):
        if not self.request_data:
            return self.LOAN_PURPOSE_OTHER
        return dict(self.LOAN_PURPOSE_CHOICES).get(self.request_data.get('loan_purpose'), self.LOAN_PURPOSE_OTHER)

    def get_monthly_revenue(self):
        if not self.response_data:
            return 0
        return json.loads(self.response_data).get('monthly_revenue', 0)

    def get_annual_revenue(self):
        if not self.response_data:
            return 0
        return json.loads(self.response_data).get('annual_revenue', 0)

    def get_rpl_result(self):
        if self.id == 32:
            print('RPL', self.request_data)
            return {}
        if not self.request_data or not self.request_data.get('RPL'):
            return {}
        print('RPL', self.request_data.get('RPL'))
        return self.request_data.get('RPL')
    
    def get_psml_result(self):
        if not self.request_data or not self.request_data.get('PSML'):
            return {}
        return self.request_data.get('PSML')
    
    def get_empcc_result(self):
        if not self.request_data or not self.request_data.get('EMPCC'):
            return {}
        return self.request_data.get('EMPCC')
    
    def get_final_result(self):
        if not self.request_data or not self.request_data.get('result'):
            return {}
        return self.request_data.get('result')
