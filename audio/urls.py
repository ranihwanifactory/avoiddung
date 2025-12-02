from django.urls import path, include

from audio.views import upload_audio

urlpatterns = [
    path('', upload_audio, name='index'),
]

