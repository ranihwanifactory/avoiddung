from django.db import models

class Audio(models.Model):
    upload = models.FileField(upload_to='uploads/')

    Jeolla = 'JL'
    Chungcheong = 'CC'
    Jeju = 'JJ'
    Gangwon = 'GW'
    Gyeongsang = 'GS'

    BIRTHPLACE_CHOICES = [
        (Jeolla, '전라도'),
        (Chungcheong, '충청도'),
        (Jeju, '제주도'),
        (Gangwon, '강원도'),
        (Gyeongsang, '경상도'),
    ]
    birthplace = models.CharField(
        max_length=2,
        choices=BIRTHPLACE_CHOICES,
        default= Chungcheong,
    )