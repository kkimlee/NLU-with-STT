# HanBert-54kN

---

### 한국어 BERT (HanBert-54kN) Model & Code Download

  * HanBERT 54kN 300만 Steps 모델의 공개를 중단합니다. (2020년 10월 25일)
  * 이전에 다운로드하여 연구와 교육에 활용하시던 비상업적 목적의 이용은 그대로 사용하셔도 되겠습니다.
  * 비상업적 목적의 활용에 대해서는 회사와 협의하여 주시기 바랍니다. info@tbai.info
  * 기업에서의 사용은 비상업적 목적으로 볼 수 없기 때문에, 라이센스 구입 문의를 해주시면 감사하겠습니다.
  * 참고 페이지 : https://twoblockai.com/resource-2/

---

### Pre-trained Hangul-BERT (HanBert-54kN)

 * HanBert-54kN  : HanBert 기본 모델 (300만 Step 학습)
   * bert_config.json
   * checkpoint    
   - model.ckpt-3000000.data-00000-of-00001  
   - model.ckpt-3000000.meta   
   - model.ckpt-3000000.index               
   - vocab_54k.txt

 * HanBert-54kN-IP  : 특허문서를 추가하여 만들어진 기본 모델 (기본 500만 + 특허 100만)
   - bert_config.json
   - checkpoint    
   - model.ckpt-6000000.data-00000-of-00001  
   - model.ckpt-6000000.meta   
   - model.ckpt-6000000.index               
   - vocab_54k.txt
  
 * HanBert-54kN-MRC  : 기계독해 학습 데이터로 Fine-tuning된 모델 (mrc_test.py에서 사용) 
   - bert_config.json
   - checkpoint    
   - model.ckpt-60000.data-00000-of-00001  
   - model.ckpt-60000.meta   
   - model.ckpt-60000.index               
   - vocab_54k.txt
 
 * usr_local_moran : 한국어 분석 라이브러리와 DB
   - libmoran4dnlp.so  
   - moran.db  
   - udict.txt  
   - uentity.txt

 * src : 
   * 구글에서 배포한 프로그램
      * https://github.com/google-research/bert
      - modeling.py  
      - optimization.py  

   * 구글에서 배포한 프로그램을 구미에 맞게 수정한 프로그램 by TwoBlock Ai
     - mrc_test.py             : 기계 독해 맛보기
     - run_korquad_1.0.py      : 기계 독해 (KorQuAD 1.0 학습 프로그램)
   
   * 구글에서 배포한 프로그램을 구미에 맞게 수정한 프로그램 by TwoBlock Ai
     - tokenization.py         : 한글 토크나이저 (moran 구동 모듈 포함)
     - tokenization_test.py    : 한글 토크나이저 테스트

   * 투블럭Ai에서 배포하는 형태소 분석 구동 프로그램 by TwoBlock Ai
     - moran.py  
     - moran_test.py  


#### Training Settings

* 학습 코퍼스 (일반 문서)
  * size : 3.5억개 문장,  113억개 형태소, 70G바이트
  * Dup  : 1 ~ 10 (코퍼스 성격에 따라 조합)
  
* 학습 코퍼스 (특허 문서)
  * size : 3.7억개 문장,  150억개 형태소, 75G바이트
  * Dup  : 2 

* 학습 환경
  * Google TPU V3-8
  * 기본 학습 Step수 : 500만 (공개 버젼은 300만)
  * 특허 추가 Step수 : 600만 (일반 500만 + 특허문서 100만)
  
* 한국어 형태소 분절
  * moran wiki 버젼 : moran-2013 버젼 형태소 분석기의 Deep NLP 버젼 
  * 품사를 표시하지 않고 음절단위로 분절 및 ~기호 부착
  * 앞의 형태소에 붙여야 하는 경우 ~, 그런데 기능어였던 경우 ~~
  * moran.db (126M) : 기본 지식 (64만단어 + 57만 기분석사전 + 174만 명칭어) + wiki 표제어 (445,701 단어)
  * 초당 0.5M 수준의 분절 속도 (형태소 분석후, 분절 + 기호 부착)

* Vocab
  * 크기 : 총 54000표제어 : 53800개 단어 + 200개의 여유 공간

#### Requirements

* Python : 3.6.8
* tensorflow-gpu : 1.11.0
* OS : ubuntu 18.04

#### How to Install

* 모델과 DB File의 크기가 4G입니다. Git 배포 용량을 초과합니다.
* 윈도우 버젼이나, ubuntu 18.04와 호환되지 않는 환경은 코드를 지원하고 있지 않습니다.
* 글의 아래부분에 있는 [다운로드]를 눌러서 hanbert54kN.tar.gz를 다운로드합니다.
* hanbert54kN.tar.gz를 풀면, 맨처음 해야 하는 일은 moran의 활성화입니다.
* moran을 사용하기 위해서는 아래와 같이 특정 디렉토리에 복사가 필요합니다.
* 복사만 하면, 그대로 사용할 수 있습니다.
* 다른 디렉토리에 사용하고 싶은 경우는 이 글의 아래부분에 설명되어 있습니다.

```
tar xvfz hanbert.tar.gz
cd HanBert-54kN/
sudo mkdir /usr/local/moran
cp usr_local_moran/* /usr/local/moran
```

* Moran의 동작여부 확인
   ```
   $ python src/tokenization_test.py
   Cat.12가 600 Mbps,
   12 ~~가 600 ~mbps
   ['cat.12', '~~가', '600', 'mbps', ',']
   cat.12 ~~가 600 mbps ,
   Cat.12가 600 Mbps
   ['나', '~~는', '걸어가', '~~고', '있', '~~는', '중', '~~입', '~~니다', '.', '나는걸어', '가', '~~고', '~있', '~~는', '중', '~~입', '~~니다', '.', '잘', '분류', '~~되', '~~기', '~~도', '한', '~~다', '.', '잘', '먹', '~~기', '~~도', '한', '~~다', '.']
   나 ~~는 걸어가 ~~고 있 ~~는 중 ~~입 ~~니다 . 나는걸어 가 ~~고 ~있 ~~는 중 ~~입 ~~니다 . 잘 분류 ~~되 ~~기 ~~도 한 ~~다 . 잘 먹 ~~기 ~~도 한 ~~다 .
   ['나', '~~는', '걸어가', '~~고', '있', '~~는', '중', '~~입', '~~니다', '.', '나', '##는걸', '##어', '가', '~~고', '~있', '~~는', '중', '~~입', '~~니다', '.', '잘', '분류', '~~되', '~~기', '~~도', '한', '~~다', '.', '잘', '먹', '~~기', '~~도', '한', '~~다', '.']
   ```

* HanBert에 대해서는 더 이상의 필요사항이 없습니다.

* 추가적인 사항들은 구글에서 공개한 https://github.com/google-research/bert 실행 환경과 동일합니다.
  - Python : 3.6.8
  - tensorflow-gpu : 1.11.0
  
  * KorQuAD 1.0 학습 (https://korquad.github.io/category/1.0_KOR.html)
  * KorQuAD 1.0의 데이터를 다운로드 받으세요. KorQuAD_v1.0_train.json, KorQuAD_v1.0_dev.json
  * 평가용 프로그램을 다운로드 받으세요. KorQuAD_v1.0_dev.json
  * 예제로 나와있는 코드를 실행해 보세요
  ```
  python src/run_korquad_1.0.py --init_checkpoint=HanBert-54kN --bert_config_file=HanBert-54kN/bert_config.json 
  --vocab_file=HanBert-54kN/vocab_54k.txt --do_train=true --do_predict=true --train_batch_size=16 
  --num_train_epochs=1.0 --learning_rate=3e-5 --train_file=korquad.1.0/KorQuAD_v1.0_train.json 
  --predict_file=korquad.1.0/KorQuAD_v1.0_dev.json --output_dir=result

  python evaluate-v1.0.py korquad.1.0/KorQuAD_v1.0_dev.json result/predictions.json

  ```

#### 기계 독해 맛보기
* GPU만 있으면 실행해 볼 수 있습니다.
* 미리 학습을 시킨 기계 독해 모델로 느껴보실 수 있도록 학습된 모델을 공개합니다.
    - Model명 : HanBert-54kN-MRC
    - epochs : 3.0
    
* 학습 데이터가 서로 다른 지침으로 작성되어서 동일한 형식의 질문에 각각 다른 답변의 유형이 존재합니다.
  * 학습데이터의 품질에 대해서도 의구심이 있지만, 공개되어 있는 학습데이터를 사용하여 학습하였습니다.
    - 문제시 되는 것들 :
    - 예 : 어디에서 출발해?   A) 부산    B) 부산에서

* 학습 데이터 
  * HanBert-54kN + 기계독해 학습 데이터 (총 253,466건)
    - NIA의 기계독해 데이터 : 162,632건
    - KorQuAD 1.0의 데이터 : 66,181건
    - KorQuAD 2.0의 데이터 : 24,653건 
  
* 실행 방법과 예제 
 * HanBert-54kN-MRC 디렉토리에 있는 bert_config와 model, vocab을 사용하세요.
 * src/mrc_test.py 코드에 기본으로 지정되어 있으나, 디렉토리가 변경된 경우에 지정해 주셔야 합니다.
 
    ```
    $ python src/mrc_test.py --example_file=korquad.1.0/KorQuAD_v1.0_dev.json

    =========== 기계독해 예문 964건 Loadind Done ...  by TBai ==============

    * 명령어  : 1) 내용보기, 2) 다음 예제, 3) 끝내기
    * 질의문  : 1
    예제 : 1989년 2월 15일 여의도 농민 폭력 시위를 주도한 혐의(폭력행위등처벌에관한법률위반)으로 지명수배되었다. 1989년 3월 12일 서울지방검찰청 공안부는 임종석의 사전구속영장을 발부받았다. 같은 해 6월 30일 평양축전에 임수경을 대표로 파견하여 국가보안법위반 혐의가 추가되었다. 경찰은 12월 18일~20일 사이 서울 경희대학교에서 임종석이 성명 발표를 추진하고 있다는 첩보를 입수했고, 12월 18일 오전 7시 40분 경 가스총과 전자봉으로 무장한 특공조 및 대공과 직원 12명 등 22명의 사복 경찰을 승용차 8대에 나누어 경희대학교에 투입했다. 1989년 12월 18일 오전 8시 15분 경 서울청량리경찰서는 호위 학생 5명과 함께 경희대학교 학생회관 건물 계단을 내려오는 임종석을 발견, 검거해 구속을 집행했다. 임종석은 청량리경찰서에서 약 1시간 동안 조사를 받은 뒤 오전 9시 50분 경 서울 장안동의 서울지방경찰청 공안분실로 인계되었다.


    * 명령어  : 1) 내용보기, 2) 다음 예제, 3) 끝내기
    * 질의문  : 몇시간 동안 조사 받았어?
    ... Reading : 1 pages Start ... Done ... 답변 신뢰도 :  (8.460, 75.38%)
    독해결과  : 1시간


    * 명령어  : 1) 내용보기, 2) 다음 예제, 3) 끝내기
    * 질의문  : 몇시에 공안분실로 인계되었어?
    ... Reading : 1 pages Start ... Done ... 답변 신뢰도 :  (15.47, 91.88%)
    독해결과  : 오전 9시 50분


    * 명령어  : 1) 내용보기, 2) 다음 예제, 3) 끝내기
    * 질의문  : 2
    다음 문서로 갑니다.
 
    * 명령어  : 1) 내용보기, 2) 다음 예제, 3) 끝내기
    * 질의문  : 1
    예제 : "내각과 장관들이 소외되고 대통령비서실의 권한이 너무 크다", "행보가 비서 본연의 역할을 벗어난다"는 의견이 제기되었다. 대표적인 예가 10차 개헌안 발표이다. 원로 헌법학자인 허영 경희대 석좌교수는 정부의 헌법개정안 준비 과정에 대해 "청와대 비서실이 아닌 국무회의 중심으로 이뤄졌어야 했다"고 지적했다. '국무회의의 심의를 거쳐야 한다'(제89조)는 헌법 규정에 충실하지 않았다는 것이다. 그러면서 "법무부 장관을 제쳐놓고 민정수석이 개정안을 설명하는 게 이해가 안 된다"고 지적했다. 민정수석은 국회의원에 대해 책임지는 법무부 장관도 아니고, 국민에 대해 책임지는 사람도 아니기 때문에 정당성이 없고, 단지 대통령의 신임이 있을 뿐이라는 것이다. 또한 국무총리 선출 방식에 대한 기자의 질문에 "문 대통령도 취임 전에 국무총리에게 실질적 권한을 주겠다고 했지만 그러지 못하고 있다. 대통령비서실장만도 못한 권한을 행사하고 있다."고 답변했다.


    * 명령어  : 1) 내용보기, 2) 다음 예제, 3) 끝내기
    * 질의문  : 경희대 석좌교수가 누구야?
    ... Reading : 1 pages Start ... Done ... 답변 신뢰도 :  (16.08, 99.84%)
     독해결과  : 허영


    * 명령어  : 1) 내용보기, 2) 다음 예제, 3) 끝내기
    * 질의문  : 국무회의의 심의를 거쳐야 한다는 것은 헌법 몇조야?
    ... Reading : 1 pages Start ... Done ... 답변 신뢰도 :  (17.55, 93.74%)
    독해결과  : 제89조


    * 명령어  : 1) 내용보기, 2) 다음 예제, 3) 끝내기
    * 질의문  : 3


        투블럭에이아이에서 제공하여 드렸습니다. https://twoblockai.com/

    ```

#### Moran (한국어 문장을 Deep NLP용 한국어 표현으로 변환) 예제

* moran.db, libmoran4dnlp.so의 기본 위치는 /usr/local/moran 입니다.
* moran.py의 소스코드에서 해당 so의 위치를 수정하실 수 있습니다. 
* moran.db는 moran에서 사용하는 사전과 문법이 합쳐진 것입니다. 항상 /usr/local/moran에 있어야 합니다.
* udict.txt는 사용자 기분서-사전입니다. 띄어쓰기 단위인 어절에 대한 분석 결과를 등록할 수 있습니다. 항상 /usr/local/moran에 있어야 합니다.
* uentity.txt는 사용자 명칭어 사전입니다. 현재버젼에서는 제공되지 않는 기능이므로, 그대로 두시면 됩니다. 항상 /usr/local/moran에 있어야 합니다.

```
python
>>> import moran
>>> moran_tokenizer = moran.MoranTokenizer()
>>> x = '한국어 BERT를 소개합니다.'
>>> moran_line = moran.text2moran(x, moran_tokenizer)
>>> print(moran_line)
['한국어', 'bert', '~~를', '소개', '~~합', '~~니다', '.']
>>> x = '<table> <tr> <td> 한국어 BERT를 소개합니다. </td> </tr> </table>'
>>> moran_line = moran.text2moran(x, moran_tokenizer)
>>> print(moran_line)
['<table>', '<tr>', '<td>', '한국어', 'bert', '~~를', '개', '~~합', '~~니다', '.', '</td>', '</tr>', '</table>']

```

---

### KorQuAD 1.0 학습과 추론

* 구글의 Code를 일부 수정하였습니다. 
* 아래와 같은 결과를 보신다면 정상적으로 작동하는 것으로 보입니다.

```
   $ python src/run_korquad_1.0.py --init_checkpoint=HanBert-54kN/HanBert-54kN --bert_config_file=HanBert-54kN/HanBert-54kN/bert_config.json --vocab_file=HanBert-54kN/HanBert-54kN/vocab_54k.txt --do_train=true --do_predict=true --train_batch_size=16 --num_train_epochs=5.0 --learning_rate=3e-5 --train_file=KorQuAD_v1.0_train.json --predict_file=KorQuAD_v1.0_dev.json --output_dir=54kN-result
   $ time python src/run_korquad_1.0.py --init_checkpoint=HanBert-54kN/HanBert-54kN-IP --bert_config_file=HanBert-54kN/HanBert-54kN-IP/bert_config.json --vocab_file=HanBert-54kN/HanBert-54kN-IP/vocab_54k.txt --do_train=true --do_predict=true --train_batch_size=16 --num_train_epochs=5.0 --learning_rate=3e-5 --train_file=KorQuAD_v1.0_train.json --predict_file=KorQuAD_v1.0_dev.json --output_dir=54kN-IP-result
   real	130m5.984s
   user	54m54.364s
   sys	28m56.004s

   $ python evaluate-v1.0.py KorQuAD_v1.0_dev.json 54kN-result/predictions.json
   EM = 83.495     F1 = 93.146 

   $ python evaluate-v1.0.py KorQuAD_v1.0_dev.json 54kN-IP-result/predictions.json
   EM = 81.902     F1 = 92.026 

   ```

### HanBert-54kN의 추가 학습

* 자체적인 코퍼스를 moran을 통해서 분석한 후에, 학습용 레코드를 만들어서 추가학습이 가능합니다.
* 다양한 도메인의 코퍼스로 학습하여 보세요.
* 특허 분야의 코퍼스로 추가학습한 모델을 소개합니다.
  - HanBert-54kN-IP


### Version History

* v.0.1 : 초기 모델 릴리즈

### Contacts

* info@tbai.info, 주식회사 투블럭에이아이, https://twoblockai.com/
* HanBert와 Moran의 기술 및 Resouce 도입과 튜닝, 현업 적용 등의 상업적 이용을 위한 문의를 환영합니다.
* HanBERT_MRC, HanBERT_NER의 라이센스에 대한 문의를 환영합니다.
* 자체 보유중인 코퍼스와 투블럭Ai가 보유한 정제된 코퍼스를 통합하여 자체적인 BERT를 개발하기 위한 문의를 환영합니다.

### HanBERT의 상업적 활용 안내 페이지
 * https://twoblockai.com/resource-2/
