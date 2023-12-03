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

if __name__ == "__main__":
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