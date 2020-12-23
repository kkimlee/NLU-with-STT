import numpy as np
import scipy.io as sio
import scipy.io.wavfile
from konlpy.tag import Okt


# 파일 읽어 오기
with open('dataset/transcript.v.1.4.txt', mode='r', encoding='utf-8') as file:
    scenario = list()
    file_name = list()
    
    while True:
        sentence = file.readline()
        
        if sentence:
            file_name.append(sentence.split('|')[0])
            scenario.append(sentence.split('|')[1])
        else:
            break


# 음성 파일 읽어오기
speech_data = list()
for file in file_name:
    print(file)
    dataset_path = 'dataset/kss/'
    _, data = sio.wavfile.read(dataset_path + file)
    
    speech_data.append(data)

# 형태소 분석기
okt = Okt()

# 형태소 추출
speech_data_morphs = list()
for data in speech_data:
    speech_data_morphs.append(okt.morphs(data))

