# Topic Classification LoRA 튜닝 문서

## 1. 프로젝트 목적

이 프로젝트는 Llama 3 계열 베이스 모델(예: 8B Instruct)에 LoRA를 적용해,
뉴스 본문을 `world / sports / business / sci_tech` 중 하나로 분류하도록 튜닝합니다.

핵심은 다음입니다.

- 베이스 모델 가중치는 고정
- LoRA 어댑터만 학습
- SFT 형식으로 분류 태스크를 instruction-following 형태로 학습

## 2. 어떤 데이터를 쓰는가

- 데이터셋: AG News (4-class)
- 원본 소스(스크립트 기본값):
  - `https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv`
  - `https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv`

원본 CSV 라벨은 1~4이며, 내부에서 아래로 매핑합니다.

- 1 -> world
- 2 -> sports
- 3 -> business
- 4 -> sci_tech

## 3. 데이터를 어떻게 가져오고 가공하는가

실행 스크립트:

- `scripts/data/prepare_topic_classification_data.py`

실제 로직 모듈:

- `src/data/topic_classification.py`

동작 순서:

1. 원본 CSV 다운로드 또는 로컬 CSV 로드
2. train split을 stratified 방식으로 train/val 분리
3. 각 샘플을 `messages` 기반 SFT 형식으로 변환
4. `train.jsonl`, `val.jsonl`, `test.jsonl`, `manifest.json` 생성

기본 출력 경로:

- `data/topic_classification/ag_news/`

생성 파일:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `manifest.json`
- (download 사용 시) `raw/train.csv`, `raw/test.csv`

## 4. 이 프로젝트가 튜닝하는 작업 정의

분류 입력은 뉴스 텍스트(제목+본문)이고, 출력은 단일 라벨 문자열입니다.

시스템 지시 예시:

- "Return exactly one label and nothing else: world, sports, business, or sci_tech."

유저 프롬프트는 기사 본문을 포함하고 마지막에 `Label:`로 끝납니다.
어시스턴트 정답은 라벨 텍스트 1개만 넣습니다.

관련 코드:

- `src/tasks/topic_classification.py`

## 5. 학습 파이프라인

엔트리포인트:

- 일반 LoRA SFT: `scripts/train_lora_sft.py`
- 주제분류 전용: `scripts/train_topic_classification_lora.py`

학습 핵심 로직:

- `src/train/lora_sft_trainer.py`

설정:

- `src/train/config.py`

데이터로더:

- `src/data/sft_dataset.py`

학습 특성:

- LoRA 타겟: 기본 `wq,wk,wv,wo`
- assistant 메시지 위치만 loss 계산
- gradient accumulation / cosine lr schedule / clip grad 사용
- 어댑터 체크포인트 저장 (`adapter_final.pt` 등)

## 6. 평가 파이프라인

엔트리포인트:

- `scripts/eval_topic_classification.py`

평가 로직:

- `src/eval/topic_classification.py`

출력 지표:

- accuracy
- macro F1
- class별 precision/recall/F1/support
- confusion matrix
- unknown prediction rate

## 7. 현재 구조

- `llama/`: 파운데이션 모델 코어(모델/토크나이저/생성/LoRA 주입)
- `src/train/`: 학습 오케스트레이션
- `src/data/`: 데이터 준비 및 SFT 데이터셋
- `src/tasks/`: 태스크 정의/라벨 처리
- `src/eval/`: 평가 로직
- `scripts/`: CLI thin wrapper
- `tests/`: 파이프라인 스모크 테스트

## 8. 실행 요약

데이터 준비:

```bash
python3 scripts/data/prepare_topic_classification_data.py --download_raw
```

학습:

```bash
torchrun --nproc_per_node 1 scripts/train_topic_classification_lora.py \
  --ckpt_dir Meta-Llama-3-8B-Instruct \
  --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model
```

평가:

```bash
torchrun --nproc_per_node 1 scripts/eval_topic_classification.py \
  --ckpt_dir Meta-Llama-3-8B-Instruct \
  --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model \
  --lora_adapter_path outputs/topic-cls-lora/adapter_final.pt \
  --eval_data data/topic_classification/ag_news/test.jsonl
```

## 9. 확인된 것과 남은 것

확인됨:

- 데이터 준비 스크립트 실행
- 단위/스모크 테스트 실행

남음:

- 실제 GPU 환경에서 학습 full run
- 학습된 어댑터의 실제 성능 수치 확보
