import time
from datetime import timedelta

from django.contrib import messages
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.shortcuts import render, redirect
from django.urls import reverse
from django.utils import timezone

from data_extractor.extractor import DataExtractor
from data_extractor.file_reader import read_file, tokenize

from .models import ScoringRequest


def index_page(request, *args, **kwargs):
    if request.method == 'POST':
        file: InMemoryUploadedFile = request.FILES.get('file')

        try:
            file_content = read_file(file)
            tokenized_file_content = tokenize(file_content)

            # Time to wait, secure the token usage overflow
            last_scoring_request = ScoringRequest.objects.filter(
                is_completed=False,
            ).last()
            wait_time = 0
            if last_scoring_request:
                wait_time_until = last_scoring_request.created_date + timedelta(minutes=1)
                now = timezone.now()
                if wait_time_until > now:
                    wait_time = (wait_time_until - now).total_seconds()

            time.sleep(wait_time)

            scoring_request = ScoringRequest.objects.create(
                tokenized_data=tokenized_file_content,
                is_completed=False,
            )

            data_extractor = DataExtractor()
            response = data_extractor.extract_data(tokenized_file_content)

            scoring_request.response_data = response.json()
            scoring_request.is_completed = True
            scoring_request.save(update_fields=['response_data', 'is_completed'])

            return redirect(reverse('ui:index_page'))
        except Exception as exc:
            messages.error(request, exc.args[0])

    return render(request, 'ui/index.html', dict())
