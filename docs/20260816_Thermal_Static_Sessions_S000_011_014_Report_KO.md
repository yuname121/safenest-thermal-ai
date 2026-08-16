# Thermal-90 정적 세션 검토 보고서

작성일: 2026-08-16 (KST)  
대상: `session_S000_011`~`session_S000_014`  
검증 위치: PC 바탕화면 `sessions/`

## 요약

모든 세션은 checksum `PASS`였지만, `session_S000_013`만 `CAPTURE_STRUCTURE_VALID_WITH_LIMITATIONS`이고 나머지는 `CAPTURE_INVALID`였다. 조각 재조립 후 frame-counter 중복·역전·누락이 발생한 세션은 학습 데이터로 사용하지 않는다.

| 세션 | source label | valid | invalid | validator | 오류 요약 |
|---|---|---:|---:|---|---|
| `S000_011` | `EMPTY` | 171 | 639 | `CAPTURE_INVALID` | gap 2254, duplicate 626, reversal 38 |
| `S000_012` | `STANDING` | 173 | 1709 | `CAPTURE_INVALID` | duplicate 1732, reversal 82 |
| `S000_013` | `SITTING` | 174 | 1 | `CAPTURE_STRUCTURE_VALID_WITH_LIMITATIONS` | frame-counter 오류 없음 |
| `S000_014` | `LYING` | 173 | 614 | `CAPTURE_INVALID` | gap 1147, duplicate 588, reversal 27 |

## 해석 및 조치

- `S000_013`은 구조 검토용 후보로 보관한다.
- `S000_011`, `012`, `014`는 삭제하지 않고 오류 증거로 보존한다.
- invalid 세션에서 유효 프레임만 추출해 새 학습 세션을 만들지 않는다.
- 재수집 대상은 `S000_015 EMPTY`, `S000_016 STANDING`, `S000_017 LYING`이다.
- 재수집 시 Pi 수집기를 먼저 시작하고 ESP32를 재시작한다.
- `send_failures` 증가와 frame-counter 오류가 반복되면 대량 수집을 중단하고 UDP chunk sequence를 포함하는 프로토콜 개선을 검토한다.

## 공통 제한

각 세션은 temporal provenance가 `TEMPORAL_ORDER_ONLY`이며, physical unit/orientation이 아직 검증되지 않았다. 따라서 정적 자세 pilot은 모델 학습 승인이나 낙상 이벤트 성능 검증을 의미하지 않는다.
