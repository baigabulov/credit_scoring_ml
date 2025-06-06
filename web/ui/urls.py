from django.urls import path

from . import views

app_name = 'ui'

urlpatterns = [
    path('', views.index_page, name='index_page'),
    path('login/', views.login_page, name='login_page'),
    path('logout/', views.logout_page, name='logout_page'),
    path('loans/', views.loans_page, name='loans_page'),
    path('stats/', views.stats_page, name='stats_page'),
    path('scoring/<int:application_id>/', views.scoring_page, name='scoring_page'),
    path('scoring/status/<int:application_id>/', views.scoring_status_page, name='scoring_status_page'),
    path('result/<int:application_id>/', views.result_page, name='result_page'),
]