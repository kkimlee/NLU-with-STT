import numpy as np
import scipy.io as sio
import scipy.io.wavfile
from konlpy.tag import Okt
from konlpy.tag import Kkma


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

'''

'''

# Twitter 형태소 분석기
okt = Okt()

# 형태소 분석
okt_morphs = list()
okt_pos = list()
okt_nouns = list()
for data in scenario:
    # 형태소 추출
    okt_morphs.append(okt.morphs(data))
    # 형태소 태깅
    okt_pos.append(okt.pos(data))
    # 명사 추출
    okt_nouns.append(okt.nouns(data))
    

# 꼬꼬마 형태소 분석기
kkma = Kkma()

# 형태소 분석
kkma_morphs = list()
kkma_pos = list()
kkma_nouns = list()
for data in scenario:
    # 형태소 추출
    kkma_morphs.append(kkma.morphs(data))
    # 형태소 태깅
    kkma_pos.append(kkma.pos(data))
    # 명사 추출
    kkma_nouns.append(kkma.nouns(data))

'''
# 음성 파일 읽어오기
speech_data = list()
for file in file_name:
    print(file)
    dataset_path = 'dataset/kss/'
    _, data = sio.wavfile.read(dataset_path + file)
    
    speech_data.append(data)
'''

