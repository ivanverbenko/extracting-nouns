# urls.py
from django.urls import path
from .views import TextProcessingView

app_name = 'tagger'

urlpatterns = [
    path('extract_nouns/', TextProcessingView.as_view(), name='extract_nouns'),
]
