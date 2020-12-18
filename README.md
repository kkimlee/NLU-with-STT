# NLU-with-STT
2020 MZ 인공지능 해커톤 대회 - AI 장치용 STT(Speech To Text)를 위한 의도 분류 (AI-NLU with STT)

## 기한
2020년 12월 21일(월) ~ 2020년 1월 5일(화)

## 목표

### 주관사가 제공하는 음성 인식 데이터 Set를 활용하여 , 우수한 의도 분류 시스템 개발

#### 의도 분류
+ 음성인식을 통과한 텍스트 데이터 세트 제공.(음성인식 오류 포함되어 있음)
+ AI 시스템에 대한 명령어 의도 정보가 레이블된 데이터
+ Baseline 시스템 구성도 및 성능 공개
+ EM을 기반으로 의도 분류 정확도 측정

#### 의도 클래스
+ 약 1,100가지 분류 

#### 제공데이터
+ 학습데이터 A-1 : 수집중인 명령어 음성 및 전사 텍스트 데이터-전체 80% 분량
+ 학습데이터 A-2 : A-1 학습데이터를 음성 인식한 결과 Text 파일
+ 베이스라인 시스템 : 음성인식 된 Text(A-2)에 대하여 미디어젠 ALBERT를 통한 intention(의도=요구서비스) 분류 fine-tuning 모델 및 분류시스템
+ 최종심사 평가용 데이터: 학습, Dev에 포함되지 않은 나머지 음성인식 결과 데이터 10%
+ 평가 Tool : EM 출력 코드

#### 참가자 결과 제출 및 심사과정
1. 기한 내 Dev용 데이터로 성능을 추출하여 최적 모델을 각자 업로드
2. 심사 평가용 데이터를 참가자에게 전달 & 각자 성능 측정(제출된 모델 수정 불가)
3. 각자 제출한 모델을 활용한 감정 분류 시스템 및 성능 측정 결과 제출
4. 심사자가 재현 및 결과가 일치됨을 확인
5. 결과 인정

## 역할 분담
김윤수 :

김민태 :

김성준 :

정규빈 :

## 참고 자료
kaggle 데이터
> https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset

딥러닝을 활용한 자연어 처리 기술 실습
> https://www.youtube.com/watch?v=F8b0jGyZ_W8 1강    
> https://www.youtube.com/watch?v=MNq9_XBQqms 2강    
> https://www.youtube.com/watch?v=I7JpGYK3Y-Y 3강    
> https://www.youtube.com/watch?v=2QzwIThP8pw&t=571s 4강

SKplanet Tacademy 자연어/음성인식/음성합성/AI스피커
> https://www.youtube.com/c/SKplanetTacademy/playlists?view=50&sort=dd&shelf_id=4

배원식, 차정원, "정서분석을 위한 의견관계 자동 추출", 한국정보과학회논문지: 소프트웨어 및 응용, 제 40권, 제 5호, pp. 473-481, 2013.
> https://www.dbpia.co.kr/pdf/pdfView.do?nodeId=NODE02223717&mark=0&useDate=&bookmarkCnt=0&ipRange=N&accessgl=Y&language=ko_KR

KoSpeech
> github    
> https://github.com/sooftware/KoSpeech    
> 논문 참조    
> https://arxiv.org/abs/2009.03092

ALBERT Classifier
> github    
> https://github.com/google-research/albert    
> 논문 참조    
> https://arxiv.org/abs/1909.11942    
> 리뷰    
> https://jeonsworld.github.io/NLP/albert/    
> https://y-rok.github.io/nlp/2019/10/23/albert.html    
> http://isukorea.com/blog/home/waylight3/446

Headline Calssification with SVM
> RAMESHBHAI, Chaudhary Jashubhai; PAULOSE, Joy. Opinion mining on newspaper headlines using SVM and NLP. International Journal of Electrical & Computer Engineering (2088-8708), 2019, 9.3.

카카오 형태소 분석기    
> https://github.com/kakao/khaiii    

자연어 처리 기술 동향
> https://www.aitimes.kr/news/articleView.html?idxno=15036
