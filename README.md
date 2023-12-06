# 2023-2학기 캡스톤디자인 최종보고서

## 1. 작품 제목
- **Everytime Text Gen: KoGPT-2 Finetuning**

## 2. 필요성

SKT-AI에서 한국어 데이터를 Pre-Training 시킨 KoGPT-2를 fine-tuning하여 에브리타임 게시글을 생성해 주는 모델을 만들었다. Everytime Text Gen: KoGPT-2 Finetuning(이하 ETG)은 에브리타임에서 사용자들이 다양한 주제에 관한 게시글을 여러 관점에서 효율적으로 생성할 수 있게 해주는 모델이다. 에브리타임은 자유게시판, 비밀게시판, 신입생 게시판, 졸업생 게시판 등 다양한 성격의 게시판을 운영하고 있으며, ETG는 이를 고려한 가치 있는 글을 생성하여 사용자에게 제공한다. 

또한, ETG는 사용자의 관심을 고려하여 공감 수나 댓글 수와 같은 지표를 활용하여 모델링함으로써 의미 없는 글 대신 더 많은 공감을 얻을 수 있는 글을 생성하여 유의미한 결과를 얻을 수 있도록 한다. 현재 ETG는 단순히 게시판의 성격에 맞는 글을 생성해 주는 모델이지만, 추가적인 데이터와 학습을 한다면 더 효율적으로 사용될 것으로 기대한다. 예를 들어, 해당 학교만의 특별한 질문과 답변을 자동으로 생성하여 사용자들이 정보를 더욱 신속하게 얻거나 소통이 촉진될 수 있을 것이다.
 
## 3. 작품 개요(각 구성 요소별 block diagram 수준 기술)
![image](https://github.com/Sameta-cani/kogpt_web/assets/83288284/54abf970-95f0-4e30-81e6-9e4164b0f936)
1. 에브리타임에서 크롤링을 통해 얻은 각 게시판의 정보(제목, 본문, 공감 수, 댓글 수, 스크랩 수)를 적절한 처리를 통해 데이터를 준비한다.
2. Pre-trained model인 KoGPT-2를 사용할 수 있도록 불러온다.
3. 2의 model에 1의 데이터를 넣고 Fine-tuning을 하여 ETG를 만든다.
4. Django Web Framework로 간단한 Demo Web을 만든다.
5. 이 Demo Web은 게시판 선택, 생성할 문장의 개수, 문장의 최대 길이를 사용자로부터 입력을 받으면 조건에 맞는 글을 생성해준다.

## 4. 구현 기술(플랫폼, 사용 프로그래밍 언어 및 기타 도구)
- 개발 환경: Visual Studio Code
- 사용 프로그래밍 언어 및 기타 도구
    - 프로그래밍 언어: Python 3.8.13
    - 웹 크롤링 및 데이터 전처리: selenium 4.15.2, pyperclip 1.8.2, pandas 1.4.3
    - Dataloader 생성 및 모델 학습: Pytorch 2.1.1+cu121, transformers 4.34.1(KoGPT-2)

## 5. 작품 해석(소스 설명, 구현 방법, 문제 사항 보완 결과 등)

## 6. 파일 구조

```📦kogpt_web
 ┣ 📂.git
 ┣ 📂config
 ┣ 📂model
 ┃ ┣ 📂data
 ┃ ┃ ┗ 📜df_final.csv
 ┃ ┣ 📜dataloader.py
 ┃ ┣ 📜eve_craw.ipynb
 ┃ ┣ 📜generate.py
 ┃ ┣ 📜train.py
 ┃ ┗ 📜utils.py
 ┣ 📂static
 ┃ ┣ 📂best_model
 ┃ ┃ ┣ 📜config.json
 ┃ ┃ ┣ 📜generation_config.json
 ┃ ┃ ┗ 📜pytorch_model.bin
 ┃ ┗ 📂css
 ┃ ┃ ┗ 📜style.css
 ┣ 📂templates
 ┃ ┗ 📜index.html
 ┣ 📜.gitignore
 ┣ 📜db.sqlite3
 ┗ 📜manage.py
```

## 7. result
![image](https://github.com/Sameta-cani/kogpt_web/assets/83288284/31d8ffbb-224b-4f6e-aa6c-8b8044295a8c)

## 8. KoGPT2
https://github.com/SKT-AI/KoGPT2
