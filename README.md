# kogpt_web

이 프로젝트는 학교 졸업 작품으로 개발되었으며, 주요 내용은 다음과 같다.

KoGPT2 모델을 에브리타임 게시글에 맞게 fine-tuning하여 해당 게시판의 특성에 부합하는 글을 생성한다.

이를 Django 웹프레임워크를 사용한 간단한 데모 버전을 제작하여 결과를 시각적으로 확인할 수 있도록 한다.

## file struct
```
📦kogpt_web
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

## result
![image](https://github.com/Sameta-cani/kogpt_web/assets/83288284/31d8ffbb-224b-4f6e-aa6c-8b8044295a8c)

## KoGPT2
https://github.com/SKT-AI/KoGPT2
