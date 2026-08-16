# Codex 작업 규칙

## 범위

- 이 저장소에서는 Thermal-90 열화상 AI의 데이터 계약, 오프라인 검증, T-C 현장/도메인 검증, 승인된 T-D 재학습만 다룬다.
- `firmware/xiao_esp32c6_thermal90_udp_capture/`와 `scripts/thermal_udp_capture.py`는 raw full-frame을 보존하기 위한 수집 계약 구현으로만 예외적으로 범위에 포함한다. 이외의 ESP 펌웨어, TCP/UDP 제품 통신, 위험도 융합, 경보 정책은 범위 밖이다.
- `HUMAN_FALL`은 자세 proxy다. 실제 낙상 탐지 성능이나 안전 보장을 주장하지 않는다.

## 데이터와 artifact

- 원시 열화상, 식별 가능 메타데이터, `.tflite`/학습 체크포인트/대용량 배열은 Git에 커밋하지 않는다.
- `wifi_secrets.h`에는 Wi-Fi 비밀번호와 Pi endpoint가 들어가므로 Git에 커밋하지 않는다. 제공된 `wifi_secrets.example.h`만 추적한다.
- `data/`와 `artifacts/`는 로컬 또는 승인된 외부 저장소용이다. Git에는 manifest, checksum, 재현 스크립트, 비식별 요약만 남긴다.
- 실제 수집 자료를 재학습에 승격하기 전, 원본 보존·동의/권한·dataset identity·group split·누수 검사를 확인하고 사람의 명시 승인을 받는다.

## 변경 절차

1. 작업 시작 전 `NEXT_STEPS_KO.md`와 관련 T-A/T-B 보고서를 읽는다.
2. 기존 단계의 validator/test를 먼저 실행하고 결과를 기록한다.
3. 코드는 `experiment/thermal-...` 브랜치에서 변경한다. `main`에 직접 기능 변경하지 않는다.
4. 모델/전처리/분할 변경은 데이터 계약, 설정, 결과 manifest, 평가표, checksum을 함께 갱신한다.
5. 커밋, push, PR, 외부 레포 생성은 사용자 명령 또는 해당 작업에 대한 명시 승인이 있을 때만 한다.

## 금지

- 구형 `thermal_fall_int8_v0.1.0.tflite` 또는 구형 min-max runtime을 현재 기준선으로 되살리지 않는다.
- source-level group split을 frame-level random split으로 바꾸지 않는다.
- TRAIN 이외 데이터로 전처리 통계를 fit하거나, LOCKED_TEST를 모델 선택에 사용하지 않는다.
- T-C 도메인 검증 증거 없이 T-D 재학습이나 배포 가능 판정을 내리지 않는다.
