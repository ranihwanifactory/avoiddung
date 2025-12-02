import time

import torch
from django.shortcuts import render

from django.views.decorators.csrf import csrf_exempt

from audio.models import Audio
from audio.mfcc_cnn import mfcc_spectrogram_prediction
from audio.pitch_lstm_ensemble import ensemble_pitch_prediction

@csrf_exempt
def upload_audio(request):
    if request.method == 'POST':
        audio = Audio()
        audio.upload = request.FILES['audio_file']
        audio.save()

        time.sleep(2)

        path = 'audio/media/' + str(audio.upload)
        result = predict_birthplace(path)
        audio.birthplace = result
        audio.save()

        context = {
            'result': audio.get_birthplace_display()
        }

        if audio.birthplace == 'GW':
            template_name = 'audio/result_gw.html'
        elif audio.birthplace == 'CC':
            template_name = 'audio/result_cc.html'
        elif audio.birthplace == 'GS':
            template_name = 'audio/result_gs.html'
        elif audio.birthplace == 'JJ':
            template_name = 'audio/result_jj.html'
        elif audio.birthplace == 'JL':
            template_name = 'audio/result_jl.html'

        audios = Audio.objects.all()
        audios.delete()

        return render(request, template_name=template_name, context=context)

    return render(request, 'audio/index.html')

def predict_birthplace(path):

    result = []

    list = []

    # Class category and index of the images: {'Chungcheong': 0, 'Gangwon': 1, 'Gyeongsang': 2, 'Jeju': 3, 'Jeolla': 4}
    temp = mfcc_spectrogram_prediction(path, 'audio/model/mfcc_spectrogram_cnn.pt')
    list.append(temp[:,1])
    list.append(temp[:,2])
    list.append(temp[:,4])
    list.append(temp[:,3])
    list.append(temp[:,0])

    cnn_result = torch.tensor([list])

    result.append(cnn_result)

    # 0: 강원도, 1: 경상도, 2: 전라도, 3: 제주도, 4: 충청도
    lstm_result = ensemble_pitch_prediction(path, 'audio/net/')[0]
    result.append(lstm_result)

    result = sum(result)

    _, output_index = torch.max(result, 1)

    if output_index == 0:
        return 'GW'
    elif output_index == 1:
        return 'GS'
    elif output_index == 2:
        return 'JL'
    elif output_index == 3:
        return 'JJ'
    elif output_index == 4:
        return 'CC'

