# KoCharELECTRA

**Character-level (음절)** Korean ELECTRA Model

## Details

Wordpiece-level이 아닌 **Character-level(음절) tokenizer**를 이용하여 학습한 한국어 ELECTRA Model입니다.

|       **Model**       | Max Len | Vocab Size |
| :-------------------: | ------: | ---------: |
| `KoCharElectra-Base`  |     512 |      11568 |
| `KoCharElectra-Small` |     512 |      11568 |

- Vocab의 사이즈는 `11568`개로 `[unused]` 토큰 200개를 추가하였습니다.
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

- **Huggingface S3**에 모델이 이미 업로드되어 있어서, **모델을 직접 다운로드할 필요 없이** 곧바로 사용할 수 있습니다.

- `ElectraModel`은 `pooled_output`을 리턴하지 않는 것을 제외하고 `BertModel`과 유사합니다.

- ELECTRA는 finetuning시에 `discriminator`를 사용합니다.

### 🚨 주의사항 🚨

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

# Output
--------------------------------------------------------
tokens:  [CLS] 나 는   걸 어 가 고   있 는   중 입 니 다 .   나 는 걸 어   가 고 있 는   중 입 니 다 . [SEP]
input_ids: 2 40 8 5 374 38 14 13 5 36 8 5 75 142 57 7 10 5 40 8 374 38 5 14 13 36 8 5 75 142 57 7 10 3 0 0 0 0 0 0
token_type_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
attention_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0
--------------------------------------------------------
[Last layer hidden state]
Size: torch.Size([1, 40, 768])
Tensor: tensor([[[-0.2509,  0.3669,  0.1397,  ...,  0.2624, -0.2964,  0.4341],
         [ 0.8062, -0.5277, -0.0232,  ...,  0.1310, -0.2489,  0.0741],
         [ 0.0750, -0.4534, -0.0410,  ..., -0.0769, -0.4361,  0.0842],
         ...,
         [ 0.3351, -0.2150, -0.2237,  ..., -0.0798, -0.2663,  1.3754],
         [ 0.2643, -0.2015, -0.1858,  ..., -0.0495, -0.2822,  1.3010],
         [ 0.1068, -0.4641, -0.1854,  ...,  0.0265, -0.3297,  1.1307]]])
```

## Result on Subtask

Char-level인 관계로 `max_seq_len`은 128인 최대 길이로 돌렸지만, KoELECTRA와 비교했을 때 나쁘지 않은 성능을 보였습니다.

### Base Model

|                        | NSMC (acc) | Naver NER (F1) |
| ---------------------- | :--------: | :------------: |
| KoELECTRA-Base         |   90.21    |     86.87      |
| **KoCharELECTRA-Base** | **90.18**  |   **84.52**    |

### Small Model

|                         | NSMC (acc) | Naver NER (F1) |
| ----------------------- | :--------: | :------------: |
| KoELECTRA-Small         |   88.76    |     84.11      |
| **KoCharELECTRA-Small** | **89.20**  |   **82.83**    |

## Acknowledgement

KoCharELECTRA은 **Tensorflow Research Cloud (TFRC)** 프로그램의 Cloud TPU 지원으로 제작되었습니다.

## Reference

- [ELECTRA](https://github.com/google-research/electra)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Tensorflow Research Cloud](https://www.tensorflow.org/tfrc?hl=ko)
