import os
import torch
import torch.nn as nn
import glob
import numpy as np
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic

import pyaudio
import wave

class RNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layers, num_classes, bidirectional=False, dropout=0.3):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb = nn.Embedding(input_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional,
                            dropout=dropout)
        if bidirectional == False:
            self.fc = nn.Linear(hidden_size, num_classes)
        else:
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # ------
        # input tensor: (batch_size, seq_length)

        x = self.emb(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1])

        out = self.activation(out)

        # y^: (batch_size, num_classes)

        return out

def ensemble_pitch_prediction(wave_file_path, models_path):
    ensenble_models = init_models(models_path)
    result = big_predict(data_preprocess(wave_file_path), ensenble_models)
    return result # 0: 강원도, 1: 경상도, 2: 전라도, 3: 제주도, 4: 충청도

def small_predict(data, imodel, device='cpu'):
    imodel.eval()
    x = data.reshape(1, -1)
    inputs = torch.from_numpy(x).type(torch.LongTensor).to(device)
    output = imodel(inputs).to(device) # (batch_size, num_classes)
    return output # tensor -> numpy

def big_predict(data, models):
    outputs = []
    for model in models:
        outputs.append(small_predict(data, model))
    output = sum(outputs)

    return outputs

def init_models(path):
    nets = glob.glob(path+'/*.pkl')
    models = []

    for net in nets:
        models.append(torch.load(net, map_location=torch.device('cpu')))
    
    return models

def normalize_pitch(pitch):
    if np.max(pitch)-np.min(pitch) == 0:
        return 0
    return pitch-np.min(pitch)

def length_adapter(values):
    values = normalize_pitch(values)
    
    if str(type(values)) == 'int':
        return None
        
    for i in range(len(values)):
        if values[i] != 0:
            break
    values = values[i:]
    for i in range(-1, -len(values), -1):
        if values[i] != 0:
            break
    values = values[:i+1]
    length = 500
    if len(values) == length:
        return values
    elif len(values) < length:
        return np.append(np.zeros(length), values)[-length:]
    else:
        return values[-length:]

def data_preprocess(path):
    # define stream chunk
    chunk = 1024

    # open a wav format music
    f = wave.open(path, "rb")
    print(f.getsampwidth(), f.getnchannels(), f.getframerate())
    # instantiate PyAudio
    p = pyaudio.PyAudio()
    # open stream
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)
    # read data
    data = f.readframes(chunk)

    # play stream
    while data:
        stream.write(data)
        data = f.readframes(chunk)

        # stop stream
    stream.stop_stream()
    stream.close()

    # close PyAudio
    p.terminate()

    signal = basic.SignalObj(path)
    return length_adapter(pYAAPT.yaapt(signal).samp_values)

