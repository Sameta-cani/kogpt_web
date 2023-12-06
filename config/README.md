### 5-5. config/gpt_generator.py

**5-5. config/gpt_generator.py**는 웹에서 "게시판의 종류", "생성할 문장의 개수", "문장의 최대 길이"를 사용자로부터 입력받아, 해당 인자에 맞는 문장을 생성하는 역할을 한다. 대부분의 코드는 **5-3-1. generate**와 같고 인자만 조금 다르다.

#### 5-5-1. initialize_model_and_tokenzier
```python
def initialize_model_and_tokenizer(model_path="static/best_model", tokenizer_name="skt/kogpt2-base-v2"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return device, model, tokenizer
```
현재 GPU의 사용 가능 여부를 확인하고, 가능하다면 "cuda"를, 그렇지 않다면 "cpu"를 선택하여 장치를 설정한다. 'model'은 앞서 **5-4-3. train**에서 학습된 best_model을 로드하고, 해당 모델을 선택된 장치로 이동시킨다. 'tokenizer'는 Hugging Face의 `AutoTokenizer()`를 사용하여 로드한다.

#### 5-5-2. generate_sentences
```python
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
```
**5-3-1. generate**와 코드가 동일하지만, 추가된 인자로 'device', 'board', 'num', 'max_len'이 있다. **5-5-1. initialize_model_and_tokenizer**에서 반환된 'device', 'model', 'tokenizer'가 `generate_sentences()`함수의 처음 3개의 인자로 사용되며, 웹에서 사용자로부터 입력받은 'board', 'num', 'max_len'이 나머지 3개의 인자로 사용되어 문장을 생성한다.

### 5-6. config/views.py

View는 사용자의 요청을 받아 처리하는 웹 사이트의 로직을 가지는 코드다.

#### 5-6-1. Import and Init
```python
from django.shortcuts import render
from .gpt_generator import initialize_model_and_tokenizer, generate_sentences

# 모델 및 토크나이저 초기화
device, model, tokenizer = initialize_model_and_tokenizer()
```
HTML 파일을 브라우저에 돌려주기 위해 `django.shortcuts.render()` 함수를 사용한다. 로직을 처리하기 위해 **5-5. config/gpt_generator.py**에서 정의한 함수들도 불러온다.

'device', 'model', 'tokenzier'를 미리 초기화하여 요청이 들어올 때마다 바로 적용할 수 있게 한다.

#### 5-6-2. index
```python
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

```
웹으로부터 들어온 요청이 "POST"라면, 입력받은 값들을 불러와 **5-5-2. generate_sentences**에 전달하여 문장을 생성한다. 이후, 생성된 문장을 "index.html" Template에 전달한다(**5-7. templates/index.html** 참고).
