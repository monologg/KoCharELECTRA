# KoCharELECTRA

**Character-level (음절)** Korean ELECTRA Model

## Details

Wordpiece-level이 아닌 **Character-level(음절) tokenizer**를 이용하여 학습한 한국어 ELECTRA Model입니다.

|       **Model**       | Max Len | Vocab Size |
| :-------------------: | ------: | ---------: |
| `KoCharElectra-Base`  |     128 |       5886 |
| `KoCharElectra-Small` |     128 |       5886 |

- Base setting은 `max_len`을 512로 설정하는데, **`KoCharElectra`에서는 `max_len`을 128로 설정하였습니다.**
- Vocab의 사이즈는 `5886`개로 `[unused]` 토큰 200개를 추가하였습니다.
- `한자`의 경우는 전처리 시에 제외되어 Vocab에 존재하지 않습니다.

## Tokenizer

- Char-level tokenizer를 위하여 `tokenization_kocharelectra.py` 파일을 새로 제작
- Transformers의 tokenization 관련 함수 지원 (`convert_tokens_to_ids`, `convert_tokens_to_string`, `encode_plus`...)

```python
>>> from tokenization_kocharelectra import KoCharElectraTokenizer
>>> tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator")
>>> tokenizer.tokenize("나는 걸어가고 있는 중입니다.")
['나', '는', ' ', '걸', '어', '가', '고', ' ', '있', '는', ' ', '중', '입', '니', '다', '.']
```

## KoCharELECTRA on Transformers

- `Transformers v2.8.0`부터 `ElectraModel`을 공식 지원합니다.

- **Huggingface S3**에 모델이 이미 업로드되어 있어서, **모델을 직접 다운로드할 필요 없이** 곧바로 사용할 수 있습니다.

- `ElectraModel`은 `pooled_output`을 리턴하지 않는 것을 제외하고 `BertModel`과 유사합니다.

- ELECTRA는 finetuning시에 `discriminator`를 사용합니다.

### 주의사항!

1. 반드시 `Transformers v2.9.1` 이상을 설치하시길 바랍니다. (**v2.9.1에서 새로 변경된 API에 맞게 tokenization 파일을 제작했습니다**)

2. tokenizer의 경우 wordpiece가 아닌 char 단위이기에 `ElectraTokenizer`가 아니라 `KoCharElectraTokenizer`를 사용해야 합니다. (레포에서 제공하고 있는 `tokenization_kocharelectra.py`를 가져와서 사용해야 합니다.)

```python
from transformers import ElectraTokenizer  # DON'T use ElectraTokenizer
from tokenization_kocharelectra import KoCharElectraTokenizer  # USE KoCharElectraTokenizer
```

```python
from transformers import ElectraModel
from tokenization_kocharelectra import KoCharElectraTokenizer

# KoCharElectra-Base
model = ElectraModel.from_pretrained("monologg/kocharelectra-base-discriminator")
tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-base-discriminator")

# KoCharElectra-Small
model = ElectraModel.from_pretrained("monologg/kocharelectra-small-discriminator")
tokenizer = KoCharElectraTokenizer.from_pretrained("monologg/kocharelectra-small-discriminator")
```

### Testing Code

```bash
$ python3 test_kocharelectra.py
```

## Result on Subtask

Char-level인 관계로 `max_seq_len`은 128인 최대 길이로 돌렸지만, KoELECTRA와 비교했을 때 나쁘지 않은 성능을 보였습니다.

### Base Model

|                        | NSMC (acc) | Naver NER (F1) |
| ---------------------- | :--------: | :------------: |
| KoELECTRA-Base         |   90.21    |     86.87      |
| **KoCharELECTRA-Base** |   90.05    |     83.57      |

### Small Model

|                         | NSMC (acc) | Naver NER (F1) |
| ----------------------- | :--------: | :------------: |
| KoELECTRA-Small         |   88.76    |     84.11      |
| **KoCharELECTRA-Small** |   88.90    |     81.97      |

## Acknowledgement

KoCharELECTRA은 **Tensorflow Research Cloud (TFRC)** 프로그램의 Cloud TPU 지원으로 제작되었습니다.

## Reference

- [ELECTRA](https://github.com/google-research/electra)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Tensorflow Research Cloud](https://www.tensorflow.org/tfrc?hl=ko)
