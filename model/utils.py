import torch
from tqdm import tqdm

def generate(input_text, tokenizer, model, num):
    sentence_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_ids = tokenizer(input_text + "|", return_tensors="pt")["input_ids"].to(device)

    for cnt in tqdm(range(num)):
        gen_ids = model.generate(
            token_ids,
            max_length=32,
            repetition_penalty=5.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            use_cache=True,
            do_sample=True,
        )

        sentence = tokenizer.decode(gen_ids[0], skip_special_tokens=True).split("|", 1)[-1].strip()
        sentence_list.append(sentence)

    return sentence_list

def calculate_normalized_weight(votes, comments, scraps, weight_voting=0.7, weight_comment=0.1, weight_scrapping=0.2, smoothing=1e-5, min_smoothing=1e-8):

    total_weight = weight_voting + weight_scrapping + weight_comment + (3 * smoothing)

    # 가중치 정규화
    normalized_weight_voting = (weight_voting + smoothing) / total_weight
    normalized_weight_scrapping = (weight_scrapping + smoothing) / total_weight
    normalized_weight_comment = (weight_comment + smoothing) / total_weight

    # 모든 항목이 0이고 smoothing도 0인 경우를 대비하여 최소 smoothing 적용
    if total_weight == 0:
        total_weight = min_smoothing

    # 각 요소에 정규화된 가중치를 적용하여 최종 가중치 계산
    normalized_weighted_score = (votes * normalized_weight_voting) + (scraps * normalized_weight_scrapping) + (comments * normalized_weight_comment)

    return normalized_weighted_score
