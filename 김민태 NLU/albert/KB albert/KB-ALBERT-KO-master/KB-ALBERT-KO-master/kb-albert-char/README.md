# KB-ALBERT-CHAR
Character-level KB-ALBERT Model and Tokenizer 

## 모델 상세 정보

### 1. Architecture
- max_seq_length=512
- embedding_size=128
- hidden_size=768
- num_hidden_layers=12
- vocab_size=23797


### 2. 학습 데이터 셋

- 일반 도메인 텍스트(위키 + 뉴스 등) : 약 25GB 
- 금융 도메인 텍스트(경제/금융 특화 뉴스 + 리포트 등) : 약 15GB

</br>

## Tokenizer
음절단위 한글 토크나이저
- 기본적으로 BertWordPieceTokenizer에서 음절만 있는 형태와 비슷
- 문장의 시작과 앞에의 띄어쓰기가 있는 음절을 제외하고는 음절 앞에 `"##"` prefix 추가
  <br>띄어쓰기 `" "`는 사전에서 제외
- Hugging Face의 Transformers 중 Tokenizer API를 활용하여 개발
- Transformers의 tokenization 관련 모든 기능 지원

```python
>>> from tokenization_kbalbert import KbAlbertCharTokenizer
>>> tokenizer = KbAlbertCharTokenizer.from_pretrained(kb_albert_model_path)
>>> tokenizer.tokenize("KB-ALBERT의 음절단위 토크나이저입니다.")
['K', '##B', '##-', '##A', '##L', '##B', '##E', '##R', '##T', '##의', '음', '##절', '##단', '##위', '토', '##크', '##나', '##이', '##저', '##입', '##니', '##다', '##.']
```

> Notes: 
> 1. Tokenizer는 `Transformers 3.0.x` 에서의 사용을 권장합니다. 
> 2. Tokenizer는 본 repo에서 제공하고 있는 `KbAlbertCharTokenizer`를 사용해야 합니다. (`tokenization_kbalbert.py`)
> 3. Tokenizer를 사용하기 위해서 별도로 제공된 모델이 저장된 경로에서 불러와야 합니다.

<br>

## How to use

### 1. Model Download

- KB-ALBERT를 사용하시고자 하시는 분들은 아래 메일로 소속, 이름, 사용용도를 간단히 작성하셔서 발송해 주세요.
- ai.kbg@kbfg.com

### 2. Source Download and Install

```shell script
git clone
cd kb-albert-char
pip install -r requirements.txt
```

### 3. Unzip model zip file
- 다음의 명령으로 디렉토리를 생성한 후, 해당 디렉토리에 메일로 제공받은 압축파일들을 해제합니다.
```
$ mkdir model
```

### 4. Using with Transformers from Hugging Face
추가 예제는 [링크](https://github.com/KB-Bank-AI/KB-ALBERT-KO/tree/master/kb-albert-char/examples) 를 참고해주시기 바랍니다.

- For PyTorch
    ```python
    from transformers import AlbertModel
    from tokenization_kbalbert import KbAlbertCharTokenizer
    
    # Load Tokenizer and Model
    tokenizer = KbAlbertCharTokenizer.from_pretrained(kb_albert_model_path)  
    pt_model = AlbertModel.from_pretrained(kb_albert_model_path)
    
    # inference text input to sentence vector of last layer
    text = '방카슈랑스는 금융의 겸업화 추세에 부응하여 금융산업의 선진화를 도모하고 금융소비자의 편익을 위하여 도입되었습니다.'
    pt_inputs = tokenizer(text, return_tensors='pt')
    pt_outputs = pt_model(**pt_inputs)[0]
    print(pt_outputs)
    # tensor([[[-0.2424, -0.1150,  0.1739,  ..., -0.1104, -0.2521, -0.2343],
    #        [-0.2398,  0.6024,  0.2140,  ..., -0.1003, -0.0811, -0.3387],
    #        [-0.0628,  0.1722, -0.2954,  ...,  0.0260, -0.1288, -0.0367],
    #        ...,
    #        [ 0.0406, -0.0463,  0.0175,  ..., -0.0016, -0.0636,  0.0402],
    #        [ 0.1111, -0.2125,  0.0141,  ...,  0.1380, -0.1252, -0.0849],
    #        [ 0.0406, -0.0463,  0.0175,  ..., -0.0016, -0.0636,  0.0402]]])
    ```

- For TensorFlow 2
    ```python
    from transformers import TFAlbertModel
    from tokenization_kbalbert import KbAlbertCharTokenizer
  
    # Load Tokenizer
    tokenizer = KbAlbertCharTokenizer.from_pretrained(kb_albert_model_path)
    
    # Load Model from pytorch checkpoint
    tf_model = TFAlbertModel.from_pretrained(kb_albert_model_path, from_pt=True)
    
    # inference text input to sentence vector of last layer
    text = '방카슈랑스는 금융의 겸업화 추세에 부응하여 금융산업의 선진화를 도모하고 금융소비자의 편익을 위하여 도입되었습니다.'
    tf_inputs = tokenizer(text, return_tensors='tf')
    tf_outputs = tf_model(tf_inputs)[0]
    print(tf_outputs)
    # tf.Tensor(
    # [[[-0.24243946 -0.11504214  0.17393401 ... -0.11044239 -0.25206116
    #    -0.23426072]
    #  [-0.2397561   0.6024074   0.2139548  ... -0.10028014 -0.08111599
    #   -0.33866274]
    #  [-0.06281117  0.17218252 -0.29536933 ...  0.02597588 -0.12882982
    #   -0.03670263]
    #  ...
    #  [ 0.04058527 -0.04625399  0.017508   ... -0.00161684 -0.06357272
    #    0.04015562]
    #  [ 0.11111088 -0.2124992   0.01409155 ...  0.13796085 -0.12516738
    #   -0.08492979]
    #  [ 0.04058535 -0.04625027  0.01748611 ... -0.0016344  -0.06360036
    #    0.04017936]]], shape=(1, 54, 768), dtype=float32)
    ```

    `tf_model.h5`가 모델 디렉토리 경로에 있는 경우 직접 불러올 수 있음 
    ```python
    # Load Model from tensorflow checkpoint
    tf_model = TFAlbertModel.from_pretrained(kb_albert_model_path)
    ```

##  Sub-tasks
|                         | NSMC (Acc) | KorQuAD (EM/F1) | 금융MRC (EM/F1) | Size |
| ----------------------- | ---------- | --------------- | -------------- | ---- |
| Bert base multi-lingual | 86.38      | 67.63 / 87.51   | 35.56 / 60.46  | 681M |
| KoBERT                  | 89.36      | 47.99 / 74.86   | 17.14 / 59.07  | 351M |
| KB-ALBERT-CHAR          | 88.49      | 75.04 / 91.08   | 74.73 / 81.50  |  44M |
    
> Note: 테스트를 진행하는 환경에 따라 결과는 다소 차이가 있을 수 있습니다.
