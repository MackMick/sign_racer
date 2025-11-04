from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('webcam/', views.webcam_view, name='webcam'),
]
