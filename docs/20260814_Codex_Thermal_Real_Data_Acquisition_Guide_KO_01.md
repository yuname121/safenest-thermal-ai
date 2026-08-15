# SafeNest 열화상 실측 수집 가이드

문서 상태: **T-C 전 사전 수집 계약 v1**
대상: 열화상 데이터를 실제로 수집하는 팀원
목적: 나중에 T-C 검증, 실제 센서와 SDT 차이 분석, 필요 시 T-D 재학습 후보 검토에 사용할 수 있는 형태로 데이터를 보존하기

이 문서는 모델을 학습하거나 실기기 성능을 판정하는 문서가 아닙니다. 첫 단계는 대량 촬영이 아니라 **짧은 장치 계약 확인용 pilot**입니다.

## 1. 먼저 기억할 한 문장

열화상에서는 사진 수보다 **원본 프레임과 그 프레임이 언제·누구의·어느 세션에서·어떤 장치로 만들어졌는지**가 더 중요합니다.

최소한 다음 계보가 끊기지 않아야 합니다.

~~~
센서가 만든 native full frame
  ├─ frame_id / sequence_index
  ├─ timestamp와 clock 종류
  ├─ subject_id / session_id / event_id
  ├─ 원본 annotation과 수정 이력
  └─ SHA-256 checksum
        ↓
실제 센서 물리 계약 확인(T-C)
        ↓
real canonical
        ↓
검증 또는 승인된 T-D 작업
~~~

62×80으로 줄인 배열, 화면 캡처, thermal_max_c 하나만 남기면 이 계보를 나중에 복구할 수 없습니다.

## 2. 현재 코드에서 확인된 것과 아직 모르는 것

현재 팀 저장소의 코드 경로를 읽기 전용으로 확인한 결과는 다음과 같습니다. 아래는 **코드에 적힌 계약**이지, 실기기에서 측정되어 T-C를 통과했다는 뜻이 아닙니다.

| 항목 | 현재 코드에서 확인된 내용 | 상태 |
|---|---|---|
| 센서 경로 | ESP32의 Thermal HAT/MI48xx 계열 코드 → SNST TCP → Raspberry Pi 수신기 | 코드 경로 확인, 실기기 미검증 |
| 프레임 | 80×62, 4,960 픽셀, TCP type 2의 big-endian uint16 payload | 코드 경로 확인, native 물리 계약 미검증 |
| 메타데이터 | frame sequence, ESP32 uptime_ms, min/max raw가 packet metadata에 있음 | 보존 여부는 수집기에서 별도 확인 필요 |
| transport | ESP32 SPI/I2C 취득 후 Wi-Fi TCP, SNST v1 | 코드 경로 확인 |
| frame rate | 코드에 설정값/분주값은 있으나 effective FPS는 미측정 | CONFIGURED_ONLY |
| 온도 단위 | 기존 수집 스크립트에 raw / 100.0가 있으나 실제 센서 단위 계약으로 검증되지 않음 | NOT_VERIFIED |
| standalone driver | sensors/thermal44는 (62,80) 처리를 예상하지만 실제 하드웨어 backend는 설치되지 않은 상태 | 실기기 연결 미검증 |
| 기존 capture 방식 | 수신한 값을 float 배열과 frame_###.npy로 저장하는 경로가 있으나 raw packet, clock, session/event provenance를 모두 보존하지 않음 | 이번 계약으로 보완 |

따라서 수집 중에는 80×62, uint16, raw / 100.0, 6.25 FPS 같은 값을 **확정 사실처럼 쓰지 말고**, 장치가 실제로 제공한 값과 확인 방법을 기록합니다.

## 3. 촬영 전 준비

각 pilot마다 아래 항목을 채운 뒤 시작합니다.

1. collection_id를 만듭니다. 예: collection_20260814_pilot01
2. 참가자는 subject_S001처럼 가명 ID만 사용합니다. 이름, 학번, 연락처를 파일명이나 manifest에 넣지 않습니다.
3. 세션마다 session_S001_001을 새로 만듭니다. 센서 재시작, 설치 변경, 장면 조건 변경이 있으면 새 세션을 만듭니다.
4. 센서 모델, hardware revision, pseudonymous device ID, firmware version, collector software version과 collector commit을 기록합니다. 모르면 UNKNOWN 또는 NOT_VERIFIED로 둡니다.
5. native width/height/dtype/raw encoding을 장치 문서나 packet에서 확인해 기록합니다. 모르는 값을 SDT 값으로 채우지 않습니다.
6. **full-frame 원본이 실제로 저장되는지** 확인합니다. 화면이 보인다는 것만으로는 충분하지 않습니다.
7. 저장 공간, 파일 생성 권한, session 폴더, frames.jsonl, annotations.jsonl 저장을 확인합니다.
8. sensor timestamp, sensor frame counter, ESP32 monotonic/uptime, Pi receive monotonic, Pi wall-clock 중 무엇을 얻을 수 있는지 확인합니다. clock마다 이름과 단위를 따로 기록합니다.
9. 안전하게 움직일 수 있는 공간과 운영자를 확인합니다. 보호되지 않은 자유 낙상 실험은 하지 않습니다.

## 4. RAW / CANONICAL / MODEL INPUT

### RAW

센서가 실제로 만든 가장 가까운 표현입니다. 가능하면 아래 두 가지를 함께 보존합니다.

- raw packet/frame bytes
- 이를 해독한 native numerical frame

RAW를 저장하기 전에 다음 처리를 하지 않습니다.

- resize, crop, rotate, flip
- per-frame min-max normalization
- global z-score
- quantization
- 색상화한 PNG/JPEG export
- 잘못된 프레임 삭제
- 모델 예측값으로 라벨 덮어쓰기

### CANONICAL

실제 센서의 encoding, 단위, native geometry, orientation, calibration을 T-C에서 확인한 뒤에 정의하는 표현입니다. 기존 SDT의 Kelvin×100, crop, bilinear, 62×80 변환을 실측 데이터에 자동 적용하지 않습니다.

### MODEL INPUT

P1 z-score, quantization 등 모델 입력은 RAW/CANONICAL에서 재생성되는 파생물입니다. 모델 입력만 저장하고 RAW를 버리지 않습니다.

## 5. 촬영 중 기록할 것

### 모든 프레임

frames.jsonl에 적어도 다음을 기록합니다.

~~~
frame_id
collection_id / subject_id / session_id
recording_id / sequence_id / event_id (해당 시)
sequence_index
sensor_frame_counter
sensor_timestamp와 unit/clock domain
device_monotonic_timestamp
host_receive_monotonic_timestamp
host_wall_time와 timezone
raw_file / decoded_native_file
byte_count / native_shape / native_dtype
raw_encoding / raw_unit_claim / unit_status
CRC·packet·packet-loss 상태
validity_status / capture_error_code / exclude_reason
annotation_status
~~~

값을 얻지 못하면 칸을 생략하지 말고 null과 상태 필드(UNKNOWN, NOT_VERIFIED, NOT_APPLICABLE)를 함께 씁니다.

### 프레임 누락·손상

깨진 프레임, partial packet, 늦은 프레임, 중복 counter, CRC 오류, decode 실패를 조용히 삭제하지 않습니다. 해당 frame row를 남기고 validity_status, capture_error_code, exclude_reason를 기록합니다. 번호를 다시 매겨 누락을 숨기지 않습니다.

### 세션과 이벤트

가능하면 끊어진 단일 사진 묶음 대신 연속 세션으로 기록합니다. 장면이 바뀌면 event ID를 새로 만들고, event를 수집한다면 다음 순서를 유지합니다.

~~~
PRE_EVENT → FALL_TRANSITION → POST_FALL_LYING → RECOVERY
~~~

단순히 LYING으로 보였다는 사실은 fall event가 아닙니다. LYING은 누워 있는 자세일 수 있고, 언제 넘어졌는지 알 수 없을 수 있습니다.

## 6. 첫 pilot에서 수집할 조건

아래는 **운영상 확인할 조건**입니다. 특정 프레임 수가 통계적으로 충분하다는 주장이 아닙니다.

| 조건 | 기록할 내용 | 모델 class와의 관계 |
|---|---|---|
| 빈 장면 | empty/background, packet continuity | EMPTY 관찰 조건; 곧바로 NOT_HUMAN 학습 승인 아님 |
| 서 있기 | 거리·시야각·복장 | source posture STANDING |
| 앉기 | 의자/가림 여부 | source posture SITTING |
| 안전하게 눕기 | 눕기 전후가 같은 세션인지 | LYING posture; fall event 아님 |
| 보통 이동 | 입장·퇴장, 천천히 움직임 | scenario condition; 자체 class 아님 |
| 부분 가림 | 가구/시야 가장자리 | hard condition; 자체 class 아님 |
| 거리·각도 변화 | 설치 높이·거리·각도 | domain condition |
| 배경 온도 변화 | 따뜻한 물체/열원은 안전한 범위에서 메모 | confounder 기록 |
| 향후 전이 | 안전 통제와 별도 승인 후에만 | event ID와 phase 필요 |

같은 사람, 같은 거리, 같은 방, 같은 옷, 같은 자세만 반복하면 실제 환경의 domain variation을 볼 수 없습니다. 반대로 위험한 상황을 무리해서 채우지도 않습니다.

## 7. 보관 폴더

계약상 의미는 다음과 같습니다. 실제 수집기는 더 적합한 폴더명을 사용할 수 있지만, 의미는 유지해야 합니다.

~~~
<collection_id>/
├── collection.json
└── subjects/<subject_id>/sessions/<session_id>/
    ├── session.json
    ├── raw/
    ├── decoded_native/
    ├── frames.jsonl
    ├── annotations.jsonl
    └── checksums.sha256
~~~

raw/가 최종화되면 immutable evidence입니다. 라벨 수정은 annotation revision으로 남기고 raw bytes를 수정하지 않습니다. 원본 파일명을 바꾸면 frames manifest와 checksum을 함께 갱신해야 하며, 촬영 후 수동 renumbering은 하지 않습니다.

## 8. 촬영 종료 후 검사

세션 폴더 또는 collection 폴더에서 다음을 실행합니다.

~~~bash
python3 scripts/validate_thermal_real_capture.py <pilot-collection-directory>
~~~

실제 폴더를 넣으면 됩니다. 결과는 다음을 확인합니다.

- manifest/JSONL 구조와 required field
- unique collection/subject/session/frame ID
- raw 파일 누락·초과·checksum mismatch
- frame count, sequence reversal/duplicate/gap
- timestamp reversal, inter-frame gap, effective FPS 요약
- packet/decode/invalid 상태
- annotation frame reference와 event range 순서
- TEMPORAL_PROVENANCE_VERIFIED, TEMPORAL_ORDER_ONLY, TEMPORAL_PROVENANCE_INSUFFICIENT 분류
- preprocessed-only와 scalar-only 구분
- subject/session/event role leakage
- REAL_LOCKED_TEST의 TRAIN 사용 또는 접근 위반

CAPTURE_STRUCTURE_VALID는 데이터가 학습 승인되었다는 뜻이 아닙니다. CAPTURE_STRUCTURE_VALID_WITH_LIMITATIONS는 수집 구조는 읽히지만 unit/FPS/시간/패킷 등 제한이 남아 있다는 뜻입니다.

## 9. 첫 pilot 전달물

첫 pilot 뒤 팀장에게 아래를 전달합니다.

1. collection.json
2. 해당 session 폴더 전체(raw/, decoded_native/, JSONL, checksum 포함)
3. validator의 JSON 결과
4. 센서 model/revision/device ID 기록
5. firmware와 collector commit
6. 설정 FPS와 validator가 계산한 effective FPS
7. packet loss, reconnect, decode failure, 누락 frame 목록
8. annotation 방법·annotator code·revision 기록
9. 모르는 항목 목록(UNKNOWN, NOT_VERIFIED)

첫 pilot을 검토하기 전에는 대량 수집, T-C, T-D, 모델 튜닝을 시작하지 않습니다.

## 10. 역할과 split

수집 단계의 역할과 나중 모델 역할은 다릅니다.

- DEVICE_CONTRACT_PILOT: 장치·파일·시간·라벨 연결 확인. 반복 열람 가능하며 pristine test가 될 수 없습니다.
- REAL_DEVELOPMENT: domain/annotation 개발 검토용.
- FUTURE_TRAIN_CANDIDATE: 나중에 승인될 수 있는 후보. 존재만으로 학습 승인 아님.
- REAL_LOCKED_TEST: 센서 계약과 프로토콜을 동결한 뒤 새 subject/session으로 수집. P1 fit, train, architecture selection, threshold/calibration, INT8 representative data, debugging에 사용하지 않습니다.

split은 frame 단위로 하지 않습니다. 가능한 강한 순서로 subject → session → event 그룹을 사용하고, 같은 subject/session/event가 여러 role에 들어가지 않게 합니다. split은 모델을 보기 전에 고정합니다.

capture contract v1에서는 DEVICE_CONTRACT_PILOT, REAL_DEVELOPMENT, FUTURE_TRAIN_CANDIDATE, REAL_LOCKED_TEST 네 role만 선언합니다. TRAIN/VALIDATION 승격이나 TRAINING_ALLOWED 권한은 이 validator가 부여하지 않으며, 이후 T-D의 별도 promotion/split evidence에서만 승인합니다.

temporal provenance를 검증하려면 같은 event_id에 대해 PRE_EVENT → FALL_TRANSITION → POST_FALL_LYING의 검증된 비중첩 phase_ranges가 있어야 합니다. frame annotation의 event_phase만 붙인 경우에는 temporal provenance 검증으로 승격되지 않습니다.

## 11. 절대 하지 말 것

- thermal_max_c 또는 max/min scalar만 저장하기
- 화면 screenshot이나 PNG/JPEG만 저장하기
- raw를 보존하기 전에 resize/normalize/quantize하기
- 나쁜 프레임을 삭제하고 번호를 다시 붙이기
- filename 순서나 filesystem mtime을 timestamp라고 부르기
- session ID 없이 촬영하기
- temporal event인데 event ID 없이 촬영하기
- LYING을 검증된 fall event로 부르기
- pilot 데이터를 pristine REAL_LOCKED_TEST로 재분류하기
- REAL_LOCKED_TEST를 train, calibration, threshold tuning에 사용하기
- frame-random train/test split하기
- 보호되지 않은 자유 낙상을 즉흥적으로 재현하기
- 기존 SDT 변환을 확인 없이 실측에 적용하기

## 12. 현장에서 보는 짧은 체크리스트

~~~
[촬영 전]
[ ] collection_id 생성
[ ] pseudonymous subject_id / session_id 생성
[ ] 센서 model/revision/device ID 확인 또는 UNKNOWN 기록
[ ] firmware / collector version·commit 기록
[ ] native geometry/dtype/encoding 기록
[ ] full-frame raw 저장 여부 확인
[ ] sensor/device/host clock과 unit 확인
[ ] 저장공간·폴더·권한 확인
[ ] 안전 통제 확인

[촬영 중]
[ ] 연속 세션 유지
[ ] frame_id와 sequence/counter 자동 기록
[ ] timestamp의 clock domain과 unit 보존
[ ] 장면/자세 변경 시 event_id 기록
[ ] 누락·중복·손상 frame을 삭제하지 않음
[ ] 파일명 수동 변경·renumbering 금지

[촬영 후]
[ ] frames.jsonl / annotations.jsonl / session.json 완성
[ ] raw와 decoded_native가 실제 참조되는지 확인
[ ] checksum 생성
[ ] validator 실행
[ ] validator 결과와 제한사항 전달
[ ] raw immutable 보관
~~~

## 13. 다음 단계

이 가이드의 다음 실제 행동은 **작은 pilot 한 세션과 validator 검토**입니다. pilot이 통과하면 T-C가 실제 sensor raw encoding, unit, geometry, orientation, effective FPS, packet integrity, ESP32→Pi 동작, real canonical 변환을 별도로 판정합니다. 이 문서만으로 실기기 검증·임상 성능·모델 성능을 주장하지 않습니다.
