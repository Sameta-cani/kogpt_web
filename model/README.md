### 5-1. model/eve_craw.ipynb

#### 5-1-1. Import
```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import pyperclip
import time
import random
import pandas as pd
import re
```
사전에 Chrome Driver를 설치하고 실행할 수 있도록 세팅했으며, selenium의 webdriver와 같이 사용하면 자동화된 소프트웨어 방식으로 해당 주소에 접근하거나 어떤 양식을 제출하는 등의 제어를 할 수 있다.

tqdm을 사용하여 크롤링 진행 상황을 확인하고, pandas로 데이터프레임을 조작하며, re를 활용하여 정규표현식으로 제어하는 등 다양한 라이브러리를 import했다.

#### 5-1-2. LogIn
```python
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 10)

# 로그인
driver.get("https://everytime.kr/login")

# 사용자 자격 증명
nid = '' # 내 ID
pyperclip.copy(nid)
driver.find_element(By.NAME, "id").send_keys(Keys.CONTROL + 'v')
npw = '' # 내 비밀번호
pyperclip.copy(npw)
secure = 'blank'
driver.find_element(By.NAME, "password").send_keys(Keys.CONTROL + 'v')
pyperclip.copy(secure)

# 로그인 양식 제출
driver.find_element(By.XPATH, '/html/body/div/div/form/input').click()

# 로그인 완료까지 대기
wait.until(EC.url_changes("https://everytime.kr/login"))
```
**5-1-1. Import**에서 언급한대로, webdriver로 Chrome에 접근한다. 코드 전반에서 자주 사용되는, `sleep()`이나 `wait()`는 에브리타임 정책상 웹 크롤링과 같은 비정상적인 접근을 방지하기 위한 조치를 우회하는 방법이다. 나의 에브리타임 ID와 비밀번호를 로그인 페이지의 해당 입력칸에 넣어주고 `click()`하여 에브리타임 웹에 로그인한다.

#### 5-1-3. Crawling
```python
# 크롤링 루프
titles, comments, votes, comment_nums, scraps = [], [], [], [], []

driver.get(f"https://everytime.kr/389113/p/1")
time.sleep(3)

iterable = range(1, 2000)
pbar = tqdm(iterable,
            total=len(iterable),
            desc='page',
            ncols=100,
            ascii=' =',
            leave=True)

for cnt in pbar: 
    # everytime 게시판 링크
    pbar.set_description(f'Current Page "{cnt}"')
    driver.get(f"https://everytime.kr/389113/p/{cnt}")

    # 페이지가 로드될 때까지 대기
    time.sleep(random.randrange(1, 7))

    # 링크 가져오기
    posts = driver.find_elements(By.CSS_SELECTOR, 'div > article > a.article')
    links = [post.get_attribute('href') for post in posts]

    # 상세 글 가져오기
    for link in tqdm(links, desc='link', position=1, leave=False):
        driver.get(link)
        # 게시판 원문 가져오기
        try:
            # 페이지가 로드될 때까지 대기
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'p.large')))
            titles.append(driver.find_element(By.CSS_SELECTOR, '#container > div.wrap.articles > article > a > h2.large').text)
            comments.append(driver.find_element(By.CSS_SELECTOR, "#container > div.wrap.articles > article > a > p.large").text)
            votes.append(driver.find_element(By.CSS_SELECTOR, '#container > div.wrap.articles > article > a > ul.status.left > li.vote').text)
            comment_nums.append(driver.find_element(By.CSS_SELECTOR, '#container > div.wrap.articles > article > a > ul.status.left > li.comment').text)
            scraps.append(driver.find_element(By.CSS_SELECTOR, '#container > div.wrap.articles > article > a > ul.status.left > li.scrap').text)
        except:
            continue
        
# 브라우저 닫기
driver.quit()

results = pd.DataFrame({'title': titles, 'main': comments, 'votes': votes, 'comment_nums': comment_nums, 'scraps': scraps})

results.to_csv('results_free.csv')
```
추출할 정보를 담을 빈 리스트 객체를 생성해 주고, 크롤링을 진행할 페이지를 설정한다. 여기서는 1페이지부터 2,000페이지까지 진행했다. 각 게시판은 고유 번호를 가지며, 자유게시판의 경우 389113에 해당한다. 설정한 페이지의 게시글에 접근하여 CSS selector를 사용하여 원하는 정보를 추출하고, 모든 페이지를 탐색한 후에는 브라우저를 닫아준다. 마지막으로, 추출한 정보를 데이터프레임에 넣어 저장한다. 참고로 1,000페이지 크롤링에는 약 6시간 22분이 소요되었다.

#### 5-1-4. Data Preprocessing
```python
def clean_str(text):
    # 여러 패턴을 하나로 통합
    patterns = [
        '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',  # E-mail 제거
        '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+',  # URL 제거
        '([ㄱ-ㅎㅏ-ㅣ]+)',  # 한글 자음, 모음 제거
        '<[^>]*>',  # HTML 태그 제거
        '[^\w\s\n]',  # 특수 기호 제거
        '[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '',  # 추가 특수 기호 제거
        '\n', '.'  # 줄 바꿈을 마침표로 변경
    ]
    combined_pattern = '|'.join(patterns)
    text = re.sub(pattern=combined_pattern, repl='', string=text)
    text = re.sub('[一-龯a-zA-Z]', '', string=text)  # 한자 및 영문 제거
    return text

def prepro_df(df):
    # 'title'이 NaN인 경우에 대비하여 fillna 사용
    df['content'] = df['title'].fillna('') + ' ' + df['main']
    df = df.drop(columns=['Unnamed: 0', 'title', 'main'])
    df = df.sample(frac=1).reset_index(drop=True)
    df['content'] = df['content'].apply(clean_str)
    return df
```
대다수의 커뮤니티에서 사용되는 언어는 구어체의 특징과 신조어의 증가, 오타, URL, 개행 문자 등의 특수 텍스트 등이 빈번하게 등장하는 특징이 있다. 또한, 영어와 한자의 빈도가 낮을 것으로 예상되어 해당 언어들을 제거하였다.

대부분의 게시글은 제목과 본문의 내용이 연결되는 경향이 있다. 예를 들어, 제목에 "오늘 밖에"가 포함된 경우, 본문에서는 "코트 입고 나가도 될 날씨야?"와 같이 제목과 본문이 연결된다(과거 특정 게시판은 제목 없이 본문만 있었으므로, 이를 고려하여 제목의 결측치를 채워주었다). 따라서, 제목과 본문을 연결한 'content'를 새로 추가해 주고 필요 없는 값들을 제거하여 전처리를 진행했다.


### 5-2. model/dataloader.py

#### 5-2-1. Import
```python
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import DataCollatorForLanguageModeling
from utils import calculate_normalized_weight
```
dataloader.py에서는 model에 데이터를 넣기 위한 Dataset과 DataLoader를 생성한다. 여기서 `WeightedRandomSampler()`와 `calculate_normalized_weight()`은 가중랜덤샘플링을 하기 위해 import했다. 이 중 `calculate_normalized_weight()`은 utils.py에서 따로 정의한 것으로, **5-3-2. calculate_normalized_weight**에서 확인할 수 있다.

#### 5-2-2. GPTDataset
```python
class GPTDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        # 데이터 불러오기 및 가중치 계산
        data = pd.read_csv(file_path)
        data['Weighted_Score'] = data.apply(lambda row: calculate_normalized_weight(row['votes'], row['comment_nums'], row['scraps']), axis=1)
        weights = torch.tensor(data['Weighted_Score'].values, dtype=torch.float32)
        weights = (weights - weights.min()) / (weights.max() - weights.min()) + 1
        
        # 토큰화
        concats = [
            label + "|" + text for label, text in zip(data["label"], data["content"])
        ]
        self.item = tokenizer(
            concats,
            return_tensors="pt", # Return PyTorch objects
            padding="max_length",
            truncation=True,
            max_length=64,
        )["input_ids"]
        self.length = len(concats)
        self.sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    def __getitem__(self, i):
        return self.item[i]
    
    def __len__(self):
        return self.length
```
Dataset은 data와 label을 저장하는 역할을 하며, 이 Dataset은 `torch.utils.data.Dataset`을 상속받아야 하고 `__init__()`, `__getitem__()`, 그리고 `__len__()`이 필수적으로 있어야 한다. `__init__()`에서는 두 가지의 역할을 한다.

첫째, 입력받은 file_path에 해당하는 데이터를 불러오고 `calculate_normalized_weight()`과 정규화로 가중치를 계산한다. 그리고, 해당 가중치를 기준으로 후에 sampling을 하기 위해 `WeightedRandomSampler()`로 감싸준다. 

둘째, 불러온 csv 파일에서 'label'과 'content'를 추출해 임의로 지정한 입력 데이터 포맷인 'label | text' 형식으로 저장한다. 예를 들어, '신입생 | 오늘 밖에 코트 입고 나가도 될 날씨야?', '자유 | 9시 달구지 출발했어?' 등으로 나타낸다. 이 값들을 미리 불러온 tokenizer로 토큰화하여 저장해둔다.

`__getitem__()`은 주어진 인덱스에 해당하는 데이터를 반환하며, `__len__()`은 전체 문장의 길이를 반환한다.

#### 5-2-3. GPTDataLoader

```python
def GPTDataLoader(tokenizer, file_path, batch_size):
    data = GPTDataset(tokenizer, file_path)
    return DataLoader(data, batch_size=batch_size, sampler=data.sampler,
                      collate_fn=DataCollatorForLanguageModeling(
                          tokenizer=tokenizer,
                          mlm=False
                      ))
```
Dataloader는 data에 쉽게 접근할 수 있도록 위에서 생성한 Dataset을 순회 가능한 객체(iterable)을 감싸주는 역할을 한다. 이 과정에서 batch_size나 sampling 방식을 제어할 수 있다. 여기서 collate_fn는 dataset이 고정된 길이가 아니고 batch_size의 값으로 2 이상인 값을 주고자 하는 경우에 직접 작성해 전달해 주어야 한다.

### 5-3. model/utils.py

##### 5-3-1. generate

```python
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
```
`generate()`는 'input_text'가 들어오면 미리 정의한 'tokenizer'와 'model'에 입력하고, 'num'개의 출력을 생성한다. 'device'는 GPU의 여부에 따라 'cuda' 또는 'cpu'로 지정되며, 'token_ids'는 앞서 **5-2-2. GPTDataset**에서 정의한 입력 포맷인 'label | content' 형식으로 만들기 위해 'input_text'에 '|'를 추가한 후 tokenizer에 적용한다. 뒤에 `.to(device)`는 Pytorch의 연산을 동일한 환경에서 하기 위한 것이다.

'num'개수만큼 반복하면서, 'model'의 `generate()`에 앞서 만든 'token_ids'와 다양한 인자들을 전달한다. 'gen_ids'는 생성된 문장에 포함된 단어의 여러 인덱스로 구성되어 있으므로, 'tokenizer'의 `decode()`에 전달하고, "|" 기준 오른쪽에 있는 텍스트를 가져온다.


#### 5-3-2. calculate_normalized_weight

```python
def calculate_normalized_weight(votes, comments, scraps, weight_voting=0.7, weight_comment=0.1, weight_scrapping=0.2, smoothing=1e-5, min_smoothing=1e-8):

    total_weight = weight_voting + weight_scrapping + weight_comment + (3 * smoothing)

    # 가중치 정규화
    normalized_weight_voting = (weight_voting + smoothing) / total_weight
    normalized_weight_scrapping = (weight_scrapping + smoothing) / total_weight
    normalized_weight_comment = (weight_comment + smoothing) / total_weight

    # 모든 항목이 0이고 smoothing도 0인 경우를 대비하여 최소 smoothing 적용
    if total_weight == 0:
        total_weight = min_smoothing

    # 각 요소에 정규화된 가중치를 적용하여 최종 가중치 계산Ready to learn
    normalized_weighted_score = (votes * normalized_weight_voting) + (scraps * normalized_weight_scrapping) + (comments * normalized_weight_comment)

    return normalized_weighted_score
```
`caluclate_normalized_weight()`는 투표, 댓글, 스크랩의 수에 가중치를 적용하고, 정규화된 가중 평가 점수를 반환한다. 사용자는 함수의 매개변수를 조절하여 각 항목의 중요도를 조절할 수 있으며, 'smoothing' 값을 사용하여 모든 항목이 0일 때를 방지한다('min_smoothing'은 'smoothing' 값이 0일 때를 대비한다).

### 5-4. model/train.py

#### 5-4-1. Import
```python
import os
import argparse
import torch
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from dataloader import GPTDataLoader
from utils import generate
```
파일 경로 처리를 위한 os 모듈, 명령행 인자를 파싱하고 처리하는 argparse, 그리고 transformers 라이브러리를 불러온다. transformers 라이브러리에서는 사용할 Model, Tokenizer, optimizer, 스케줄러의 모듈을 불러오며, 마지막 2행은 **5-2. model/DataLoader**와 **5-3-1. generate**를 참고한다.

#### 5-4-2. setting
```python
# Argument Parser 초기화
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="skt/kogpt2-base-v2", type=str)
parser.add_argument("--data_dir", default="./data", type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--warmup_steps", default=200, type=int)
args = parser.parse_args()

# 기본 디렉토리 및 데이터 디렉토리 설정
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, args.data_dir)

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.add_special_tokens({"pad_token": "<pad>"})

# 데이터셋 로드
train_dataloader = GPTDataLoader(
    tokenizer, os.path.join(DATA_DIR, "df_final.csv"), args.batch_size
)

# 모델 로드 및 학습 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained(args.model_name).to(device)
model.train()

# 옵티마이저, 스케줄러 설정
optimizer = AdamW(model.parameters(), lr=args.lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=-1
)

# 초기 손실값 설정
min_loss = float('inf')
```
`argparse` 모듈을 사용하여 명령행에서 전달되는 인자들을 처리하는 parser를 초기화한다. 이를 통해 학습에 필요한 다양한 하이퍼파라미터를 설정할 수 있다.

현재 작업 디렉토리와 데이터 디렉토리를 설정하고, Hugging Face에서 지정한 모델의 tokenizer를 로드한다. 여기에 "\<pad\>"와 같은 특수 토큰을 추가한다. 그리고 지정된 데이터 파일에서 학습에 사용할 데이터를 로드한다.

KoGPT-2 언어 모델을 불러와 지정된 device로 이동한 후 학습 모드로 설정한다. 이후, AdamW 옵티마이저를 초기화하고, 학습률과 스케줄러를 설정한다. 여기서 `get_linear_schedule_with_warmup()`은 'learning rate'를 linear하게 감소시키는 역할을 한다.

마지막으로, 초기 손실값을 무한대로 설정한다. 이는 나중에 현재 손실값과 비교하여 모델 성능을 평가할 때 사용하기 위함이다.

#### 5-4-3. train
```python
# 에폭 반복
for epoch in range(args.epochs):
    print(f"Training epoch {epoch}")

    # 데이터로더 반복
    for input_text in tqdm(train_dataloader):
        input_tensor = input_text['input_ids'].to(device)
        outputs = model(input_tensor, labels=input_tensor)
        loss = outputs.loss

        # 역전파 및 옵티마이저 업데이트
        optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # 현재 에폭의 손실 출력
    print(f"epoch {epoch} loss {outputs[0].item():0.2f}")

    # 에폭 종료 시 예제 생성
    labels = ["자유", "비밀", "신입생"]
    for label in labels:
        gen = generate(label, tokenizer, model, 1)
        print(f"{label}: {gen[0]}")

    # 각 에폭마다 모델 저장
    model_save_path = os.path.join(BASE_DIR, f"best_model_epoch{epoch}")
    model.save_pretrained(model_save_path)

    # 손실이 최소일 때 모델 저장
    if loss.item() < min_loss:
        min_loss = outputs[0].item()
        best_model_save_path = os.path.join(BASE_DIR, "best_model")
        model.save_pretrained("./best_model")

print("Training Done!")
```
지정된 에폭 수 동안 KoGPT-2 언어 모델을 학습한다. 각 에폭에서는 데이터로더에서 가져온 텍스트를 사용하여 모델을 학습하고, 손실을 역전파하여 모델을 업데이트한다. 에폭이 끝날 때마다 손실을 출력하고, `generate()` 함수를 활용하여 몇 가지 주어진 레이블에 대한 텍스트를 생성하고 출력한다.

각 에폭이 종료될 때마다 모델의 상태를 저장하고, 현재 에폭의 손실값이 최소일 때 모델의 상태를 "best_model"이라는 이름으로 기본 디렉토리에 따로 저장한다. Django Web에 적용하기 위해 "static/best_model"로 저장했다. 저장된 모델은 "config.json", "generation_config.json", "pytorch_model.bin" 파일로 구성된다.

**5-4-2. setting**에서 지정한 하이퍼파라미터로 모델을 학습했을 때 약 30시간 정도 소요되었다.
