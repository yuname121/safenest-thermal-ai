# Thermal-90 실제 Pilot 검토 보고서

작성일: 2026-08-16 (KST)  
대상 세션: `session_S000_004`  
역할: `DEVICE_CONTRACT_PILOT`

## 보존 및 무결성

- `raw_chunks/`: 904개 UDP 조각
- 조각 크기: 1320 bytes 129개, 1460 bytes 775개
- `raw/`: 130개 (`VALID` 129개, `PARTIAL` 1개)
- `decoded_native/`: 129개
- `annotations.jsonl`: 129개 (`EMPTY`)
- `frames.jsonl`: 130개
- 체크섬: `PASS`
- sensor counter gap / packet loss: 0

## Validator 결과

- `capture_status`: `CAPTURE_STRUCTURE_VALID_WITH_LIMITATIONS`
- `raw_evidence_classification`: `FULL_FRAME_RAW`
- `raw_integrity_status`: `PASS_WITH_LIMITATIONS`
- annotation coverage: `1.0`
- valid frame count: `129`
- invalid frame count: `1` (capture 종료 시 남은 미완성 조각)
- temporal provenance: `TEMPORAL_ORDER_ONLY`
- model use eligibility: `NOT_AUTHORIZED_BY_CAPTURE_VALIDATOR`

## 제한사항

- effective FPS: 약 `4.3173 FPS`
- configured FPS: `7 FPS`
- physical unit: `NOT_VERIFIED`
- orientation: `UNKNOWN_NOT_VERIFIED`
- device timestamp: 없음; Pi host timestamp만 기록
- 2초 이상 inter-frame timing gap 경고: 4회
- event ID와 `PRE_EVENT → FALL_TRANSITION → POST_FALL_LYING` phase range 없음

## 판정

UDP 조각 보존·재조립, native frame 생성, annotation coverage, checksum을 확인한 pilot으로는 유효하다. 그러나 FPS·물리 단위·방향·시간적 이벤트 provenance가 검증되지 않았으므로 이 세션을 재학습, 낙상 이벤트 성능 주장, `REAL_LOCKED_TEST` 또는 최종 모델 검증에 사용하지 않는다.

## 다음 작업

1. timing gap 및 실제 FPS 원인을 T-C에서 확인한다.
2. Thermal-90 unit, byte order, orientation을 기준 장면으로 검증한다.
3. 새 session ID로 `EMPTY`, `STANDING`, `SITTING`, `LYING` 정적 세션을 추가 수집한다.
4. 시간적 이벤트는 안전 통제, 동의, `event_id`, 비중첩 phase range를 설계한 뒤 별도 수집한다.
5. T-C 검토와 split/annotation 승인이 끝나기 전에는 재학습하지 않는다.

원본 세션과 PC 검증 JSON은 Git에 넣지 않고 외부 보존 위치에 유지한다. 이 보고서는 그 결과를 재현할 수 있는 요약 기록이다.
