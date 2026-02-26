# Llama Tune Docs

이 문서는 현재 레포의 `주제 분류(Topic Classification) LoRA 튜닝` 프로젝트를 설명합니다.

- 상세 문서: `docs/topic_classification_lora.md`

## 현재 확인 상태

- 확인 완료
  - 데이터 준비 스크립트 동작 (`scripts/data/prepare_topic_classification_data.py`)
  - 구조/파이프라인 단위 테스트 (`python3 -m unittest discover -s tests -v`)
- 미확인
  - 실제 GPU 학습 1 epoch 이상 실행 결과
  - 실제 모델 추론 기반 평가 스크립트 full run

즉, 코드 구조/데이터 파이프라인은 동작 확인됐고, 모델 학습/평가는 실행 환경(토치/토크나이저/체크포인트/GPU)에서 별도 실행이 필요합니다.
