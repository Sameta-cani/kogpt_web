# views.py
from django.shortcuts import render
from .gpt_generator import initialize_model_and_tokenizer, generate_sentences

# 모델 및 토크나이저 초기화
device, model, tokenizer= initialize_model_and_tokenizer()

def index(request):
    if request.method == 'POST':
        board = request.POST.get('board')
        num = int(request.POST.get('gen_num'))
        max_len = int(request.POST.get('mxl'))

        # generate_sentences 함수를 사용하여 문장 생성
        sentence_list = generate_sentences(model, tokenizer, device, board, num, max_len)

        sentence_list = sentence_list if sentence_list is not None else []

        context = {
            "gen_text": sentence_list,
        }
        return render(request, "index.html", context)
    else:
        return render(request, "index.html")
