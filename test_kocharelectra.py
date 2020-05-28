import argparse

import torch
from transformers import ElectraModel
from tokenization_kocharelectra import KoCharElectraTokenizer

# Get the model path
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default="monologg/kocharelectra-base-discriminator",
                    type=str, help="Path to pre-trained model or shortcut name")
args = parser.parse_args()

# Load model and tokenizer
model = ElectraModel.from_pretrained(args.model_name_or_path)
tokenizer = KoCharElectraTokenizer.from_pretrained(args.model_name_or_path)

text_a = "나는 걸어가고 있는 중입니다."
text_b = "나는 밥을 먹고 있는 중입니다."

inputs = tokenizer.encode_plus(
    text=text_a,
    text_pair=text_b,
    add_special_tokens=True,  # This add [CLS] on front, [SEP] at last
    pad_to_max_length=True,
    max_length=40
)

tokens = tokenizer.tokenize("[CLS] " + text_a + " [SEP] " + text_b + " [SEP]")

print("--------------------------------------------------------")
print("tokens: ", " ".join(tokens))
print("input_ids: {}".format(" ".join([str(x) for x in inputs['input_ids']])))
print("token_type_ids: {}".format(" ".join([str(x) for x in inputs['token_type_ids']])))
print("attention_mask: {}".format(" ".join([str(x) for x in inputs['attention_mask']])))
print("--------------------------------------------------------")

# Make the input with batch size 1
input_ids = torch.LongTensor(inputs['input_ids']).unsqueeze(0)
token_type_ids = torch.LongTensor(inputs['token_type_ids']).unsqueeze(0)
attention_mask = torch.LongTensor(inputs['attention_mask']).unsqueeze(0)

with torch.no_grad():
    output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
last_layer_hidden_state = output[0]
print("[Last layer hidden state]")
print("Size:", last_layer_hidden_state.size())
print("Tensor:", last_layer_hidden_state)
