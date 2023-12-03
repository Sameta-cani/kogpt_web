# gpt_generator.py
from transformers import GPT2LMHeadModel, AutoTokenizer
import torch

def initialize_model_and_tokenizer(model_path="static/best_model", tokenizer_name="skt/kogpt2-base-v2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return device, model, tokenizer

def generate_sentences(model, tokenizer, device, board, num, max_len):
    sentence_list = []

    token_ids = tokenizer(board + "|", return_tensors="pt")["input_ids"].to(device)

    gen_ids = model.generate(
        token_ids,
        max_length=max_len,
        repetition_penalty=5.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        use_cache=True,
        do_sample=True,
        num_return_sequences=num,
    )

    for gen_id in gen_ids:
        sentence = tokenizer.decode(gen_id, skip_special_tokens=True).split("|", 1)[-1].strip()
        sentence_list.append(sentence)

    return sentence_list
