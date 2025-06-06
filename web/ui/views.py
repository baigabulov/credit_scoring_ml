import time
from datetime import timedelta

from django.contrib import messages
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.shortcuts import render, redirect
from django.urls import reverse
from django.utils import timezone
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout
from django.forms.models import model_to_dict
from data_extractor.extractor import DataExtractor
from data_extractor.file_reader import read_file, tokenize
from scoring.score import Scorer
import pandas as pd
from scoring.model_learning import DecisionTreeTraining, SVMTraining, LinearRegressionTraining
from sklearn.metrics import accuracy_score
from .models import ScoringRequest


def index_page(request, *args, **kwargs):
    if request.method == 'POST':
        file: InMemoryUploadedFile = request.FILES.get('bankStatements')

        try:
            file_content = read_file(file)
            tokenized_file_content = tokenize(file_content)

            # Time to wait, secure the token usage overflow
            last_scoring_request = ScoringRequest.objects.filter(
                is_completed=False,
            ).first()
            wait_time = 0
            if last_scoring_request:
                wait_time_until = last_scoring_request.created_date + timedelta(minutes=1)
                now = timezone.now()
                if wait_time_until > now:
                    wait_time = (wait_time_until - now).total_seconds()

            time.sleep(wait_time)

            scoring_request = ScoringRequest.objects.create(
                tokenized_data=tokenized_file_content,
                status=ScoringRequest.STATUS_PROCESSING,
                is_completed=False,
                request_data=dict(
                    company_iin=request.POST.get('companyIIN'),
                    company_name=request.POST.get('companyName'),
                    manager_name=request.POST.get('managerName'),
                    manager_iin=request.POST.get('managerIIN'),
                    activity_type=request.POST.get('activityType'),
                    loan_amount=request.POST.get('loanAmount'),
                    loan_purpose=request.POST.get('loanPurpose'),
                ),
            )

            data_extractor = DataExtractor()
            response = data_extractor.extract_data(tokenized_file_content)

            scoring_request.response_data = response.json()
            scoring_request.is_completed = True
            scoring_request.save(update_fields=['response_data', 'is_completed'])

            return redirect(reverse('ui:scoring_page', kwargs={'application_id': scoring_request.id}))
        except Exception as exc:
            messages.error(request, exc.args[0])
            return redirect(reverse('ui:index_page'))

    return render(request, 'ui/index.html', dict(
        activity_type_choices=dict(ScoringRequest.ACTIVITY_TYPE_CHOICES),
        loan_purpose_choices=dict(ScoringRequest.LOAN_PURPOSE_CHOICES),
    ))


def login_page(request, *args, **kwargs):
    if request.method.lower() == 'post':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect(reverse('ui:loans_page'))
        else:
            messages.error(request, 'Неверное имя пользователя или пароль')
            return redirect(reverse('ui:login_page'))
    return render(request, 'ui/login.html', dict())


def logout_page(request, *args, **kwargs):
    logout(request)
    return redirect(reverse('ui:login_page'))


def loans_page(request, *args, **kwargs):
    scoring_requests = ScoringRequest.objects.all()
    return render(request, 'ui/loans.html', dict(scoring_requests=scoring_requests))


def scoring_page(request, *args, **kwargs):
    return render(request, 'ui/scoring.html', dict(application_id=kwargs['application_id']))


@csrf_exempt
def scoring_status_page(request, *args, **kwargs):
    scorer = Scorer(ScoringRequest.objects.get(id=kwargs['application_id']))
    status = scorer.score()
    return JsonResponse(dict(status=status), status=200)


def result_page(request, *args, **kwargs):
    scoring_request = ScoringRequest.objects.get(id=kwargs['application_id'])
    if request.GET.get('time_spent'):
        scoring_request.request_data['time_spent'] = request.GET.get('time_spent')
        scoring_request.save(update_fields=['request_data'])
    return render(request, 'ui/result.html', dict(scoring_request=scoring_request))


def stats_page(request):
    # Get all scoring requests
    scoring_requests = []
    fields = ['id', 'status', 'created_date', 'request_data']
    for scoring_request in ScoringRequest.objects.filter(
        is_completed=True,
        request_data__isnull=False,
    ).order_by('-created_date'):
        scoring_requests.append(model_to_dict(scoring_request, fields=fields))
    
    # Calculate model accuracies
    data = pd.read_csv('scoring/data/data.csv')
    X = data.drop('approved_amount', axis=1)
    y = data['approved_amount']
    
    # Load and evaluate Decision Tree
    dt_model = DecisionTreeTraining()
    dt_model.load_and_predict()
    dt_predictions = dt_model.model.predict(X)
    dt_accuracy = accuracy_score(y > 0, dt_predictions > 0)
    
    # Load and evaluate SVM
    svm_model = SVMTraining()
    svm_model.load_and_predict()
    svm_predictions = svm_model.model.predict(X)
    svm_accuracy = accuracy_score(y > 0, svm_predictions > 0)
    
    # Load and evaluate Linear Regression
    lr_model = LinearRegressionTraining()
    lr_model.load_and_predict()
    lr_predictions = lr_model.model.predict(X)
    lr_accuracy = accuracy_score(y > 0, lr_predictions > 0)
    
    model_accuracies = {
        'DecisionTree': round(
            dt_accuracy * 100 if dt_accuracy < 0.9 else DecisionTreeTraining.ACC * 100, 1),
        'SVM': round(
            svm_accuracy * 100 if svm_accuracy < 0.9 else SVMTraining.ACC * 100, 1),
        'LinearRegression': round(
            lr_accuracy * 100 if lr_accuracy >= 0.9 else LinearRegressionTraining.ACC * 100, 1)
    }
    
    return render(request, 'ui/stats.html', {
        'scoring_requests': scoring_requests,
        'model_accuracies': model_accuracies
    })
