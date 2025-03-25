from django.db import models


class ScoringRequest(models.Model):

    tokenized_data = models.TextField()
    response_data = models.JSONField(null=True, blank=True)
    is_completed = models.BooleanField(default=False)
    created_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_date']
        verbose_name = 'Timestamp for request'
        verbose_name_plural = 'Timestamps for requests'
