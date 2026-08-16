# SafeNest 열화상 온디바이스 AI 인수인계서

작성일: 2026-08-16 (KST)
저장소: `rla1729/safenest-thermal-ai`
범위: Thermal-90 열화상 데이터 수집 계약, 모델 검증, 재학습 준비
범위 밖: TCP/UDP 제품 통신 설계, ESP32 애플리케이션 기능, 위험도 융합, 경보 정책

## 문서 사용법

이 문서는 다음 작업자가 장비와 저장소를 처음 받아도 현재 상태를 오해하지 않고 이어서 작업하도록 작성했다.

1. `현재 하고 있는 것`을 읽어 현재 단계와 중단선을 확인한다.
2. `지금까지 한 것`에서 이미 있는 코드·근거·검증 결과를 확인한다.
3. `앞으로 해야 하는 것`을 위에서부터 순서대로 수행한다.
4. 원시 데이터, Wi-Fi 비밀번호, 모델 binary는 Git에 올리지 않는다.

## 현재 하고 있는 것

현재 단계는 **T-C 전 단계의 DEVICE_CONTRACT_PILOT 수집기 준비**다. 아직 실제 Thermal-90 센서에서 수집한 결과로 모델 성능을 판정하거나 재학습하는 단계가 아니다.

현재 하드웨어 경로:

```text
Thermal-90 → XIAO-ESP32C6 → UDP raw datagram → Raspberry Pi 수집기
                                             ↓
                                  raw/native/provenance/checksum
                                             ↓
                                  PC validator 및 T-C 검토
```

현재 논리 frame 계약은 팀 PC의 `Desktop\Thermal_Test`를 기준으로 한다. 실제 XIAO/Pi pilot에서 구형 조각을 단순 연결할 때 frame 경계가 무너진 증거가 반복되어, 새 수집용 송신기·수집기는 `SNTR` framed UDP V2를 사용한다. 각 조각은 frame ID, chunk index/count, offset/length와 전체 frame CRC32를 포함한다. 아래의 과거 `S000_004`, `S000_011`~`014` 결과는 구형 V1 수집 증거로 그대로 보존한다.

- 논리 프레임 크기: 정확히 10,080 bytes
- 새 V2 UDP 조각: 최대 1,200 bytes, frame당 9개
- V2 header: 32 bytes, network byte order, magic `SNTR`, version `2`
- 재조립 모드: `--reassemble-udp-chunks` (frame별 fail-closed + CRC32)
- 구형 blind stream: `--legacy-stream-reassembly` (진단 보존 전용, 새 수집 금지)
- 5,040 little-endian `uint16` word
- word `0..79`: 센서 header
- word `80..5039`: `80×62` pixel payload
- header word `0`: 수신 frame counter로 기록
- 물리 온도 단위, orientation, 실제 FPS는 아직 검증하지 않음

수집기는 다음을 저장한다.

```text
<collection_id>/
├── collection.json
└── subjects/<subject_id>/sessions/<session_id>/
    ├── raw/*.udp.bin
    ├── raw_chunks/*.bin               # 조각 재조립 모드에서만 생성
    ├── decoded_native/*_pixels_u16le.bin
    ├── frames.jsonl
    ├── annotations.jsonl
    ├── session.json
    └── checksums.sha256
```

수집 중에는 resize, crop, rotate, normalize, calibration, color-map, model inference를 하지 않는다. `HUMAN_FALL`은 실제 낙상 event가 아니라 LYING 유래 자세 proxy이므로, 안전하지 않은 자유 낙상 실험을 하지 않는다.

## 지금까지 한 것

### 저장소와 문서

- 공개 저장소 `rla1729/safenest-thermal-ai`를 생성했다.
- T-A0~T-B5 열화상 데이터 계약·검증 스크립트·manifest·보고서를 선별 이관했다.
- 통신/통합 노드/위험도 융합/구형 runtime과 구형 모델 binary는 현재 열화상 AI 기준선에서 제외했다.
- 실행 매뉴얼은 `NEXT_STEPS_KO.md`다.
- 실제 수집 계약은 `docs/20260814_Codex_Thermal_Real_Data_Acquisition_Guide_KO_01.md`다.
- XIAO와 Pi 수집 절차는 `docs/THERMAL90_UDP_CAPTURE_SETUP_KO.md`다.

### 새 수집 구현

- `firmware/xiao_esp32c6_thermal90_udp_capture/xiao_esp32c6_thermal90_udp_capture.ino`
  - Thermal_Test의 XIAO ESP32C6 핀·I2C·SPI 초기화 흐름을 기반으로 raw frame을 전송한다.
  - Wi-Fi와 Pi 주소는 `wifi_secrets.h`에서 읽으며 이 파일은 Git에 올리지 않는다.
- `scripts/thermal_udp_capture.py`
  - Raspberry Pi 표준 Python만으로 동작한다.
  - exact-size V1 진단 모드와 `--reassemble-udp-chunks` framed UDP V2 모드를 지원한다.
  - V2 재조립은 frame ID/chunk index/count/offset/length를 검증하고 전체 frame CRC32가 맞을 때만 10,080-byte 논리 프레임을 `raw/`와 `decoded_native/`에 기록한다.
  - 손실·중복 충돌·timeout·CRC 실패는 다음 frame bytes로 보충하지 않고 fail-closed metric으로 남긴다.
  - unexpected datagram, header counter gap/duplicate를 기록한다.
  - Pi host monotonic timestamp와 wall-clock을 기록한다.
  - 모델 입력을 만들거나 예측하지 않는다.
- `scripts/validate_thermal_real_capture.py`
  - 수집 구조·manifest·파일 존재·checksum·sequence/timestamp·annotation을 검사한다.
  - 성공해도 학습·T-C·T-D·LOCKED_TEST 사용을 승인하지 않는다.

### 검증 결과

실제 센서 연결 전 localhost UDP 시뮬레이션으로 다음을 확인했다.

- 10,080-byte frame 3개 수신
- 의도적인 header counter gap 1개를 missing marker로 보존
- raw/native/JSONL/session/checksum 생성
- validator 결과 `CAPTURE_STRUCTURE_VALID_WITH_LIMITATIONS`
- checksum `PASS`
- validator error `0`

이 결과는 코드 흐름 검증일 뿐 실제 XIAO·Thermal-90·Pi 하드웨어 검증이 아니다.

### 실제 하드웨어 pilot: `session_S000_004`

- PC 보존 위치: `Desktop\session_S000_004`
- validator 결과: `CAPTURE_STRUCTURE_VALID_WITH_LIMITATIONS`
- checksum: `PASS`
- `raw_chunks/`: 904개 (1320 bytes 129개, 1460 bytes 775개)
- 논리 프레임: `VALID` 129개, 마지막 `PARTIAL` 1개
- `decoded_native/`: 129개
- `annotations.jsonl`: 129개, 모두 `EMPTY`
- sensor counter gap / packet loss: 0
- raw evidence: `FULL_FRAME_RAW`
- 측정 effective FPS: 약 4.3173 FPS (설정값 7 FPS와 차이)
- physical unit/orientation: 아직 `NOT_VERIFIED`
- temporal provenance: `TEMPORAL_ORDER_ONLY`
- 2초 이상 inter-frame timing gap 경고: 4회
- 모델 사용 eligibility: validator가 승인하지 않음

세부 기록은 PC 바탕화면의 `pilot_review_session_S000_004_KO.md`와 `validation_session_S000_004.json`에 있다. 이 세션은 구조·전송 pilot 증거로 보관하며, 재학습·낙상 이벤트 주장·LOCKED_TEST 승격에 사용하지 않는다.

### 정적 자세 수집 결과: `session_S000_011`~`014`

PC 바탕화면 `sessions/` 아래의 네 세션을 PC validator로 재검증했다. 모든 세션의 checksum은 `PASS`였지만, 조각 재조립 후 header frame-counter 무결성이 세션별로 달랐다.

| 세션 | 라벨 | validator | 유효/무효 | 주요 결과 |
|---|---|---|---:|---|
| `session_S000_011` | `EMPTY` | `CAPTURE_INVALID` | 171 / 639 | sensor counter gap 2254, duplicate 626, reversal 38 |
| `session_S000_012` | `STANDING` | `CAPTURE_INVALID` | 173 / 1709 | duplicate 1732, reversal 82 |
| `session_S000_013` | `SITTING` | `CAPTURE_STRUCTURE_VALID_WITH_LIMITATIONS` | 174 / 1 | counter gap/duplicate/reversal 0 |
| `session_S000_014` | `LYING` | `CAPTURE_INVALID` | 173 / 614 | sensor counter gap 1147, duplicate 588, reversal 27 |

해석:

- `S000_013`만 현재 pilot 구조 검토 후보로 보관한다.
- `S000_011`, `012`, `014`는 UDP 조각 손실 또는 재조립 stream desynchronization 증거가 있어 학습·정적 라벨 검증에 사용하지 않는다.
- invalid 세션을 파일 편집으로 복구하거나 유효 프레임만 골라 새 세션으로 만들지 않는다. 원본은 오류 증거로 보존한다.
- validator JSON은 PC 바탕화면 `sessions/validation_session_S000_011.json`~`014.json`에 생성했다.

세부 요약은 `docs/20260816_Thermal_Static_Sessions_S000_011_014_Report_KO.md`에 있다.

## 앞으로 해야 하는 것

### 1. PC에서 비밀 설정 파일 생성

현재 추적되는 것은 `wifi_secrets.example.h`뿐이다. 실제 Wi-Fi 정보를 로컬 파일에 복사해 입력한다.

PowerShell:

```powershell
$fw = 'C:\Users\KIM TAEGYUN\Documents\ChatGPT\Embedded_SW\safenest-thermal-ai\firmware\xiao_esp32c6_thermal90_udp_capture'
Copy-Item -LiteralPath (Join-Path $fw 'wifi_secrets.example.h') -Destination (Join-Path $fw 'wifi_secrets.h')
code (Join-Path $fw 'wifi_secrets.h')
```

입력할 값은 다음 세 가지다.

```cpp
#define THERMAL_WIFI_SSID "실제_2G_WiFi_SSID"
#define THERMAL_WIFI_PASSWORD "실제_WiFi_비밀번호"
#define THERMAL_RECEIVER_IP "라즈베리파이_WLAN_IP"
```

`wifi_secrets.h`는 `.gitignore`에 의해 추적되지 않는다. 커밋 전에 `git status`로 비추적 파일에 나타나지 않는지 확인한다.

### 2. XIAO-ESP32C6 업로드

Arduino IDE에서 다음 폴더의 `.ino`를 연다.

```text
firmware/xiao_esp32c6_thermal90_udp_capture/
```

XIAO ESP32C6 보드와 실제 serial port를 선택해 업로드한다. 배선은 기존 Thermal_Test와 대조한다.

- SDA: D4
- SCL: D5
- MOSI: D10
- MISO: D9
- SCK: D8
- CS: D3
- DATA_READY: D1
- NRESET: D2

Serial Monitor는 `115200` baud로 열고 다음을 확인한다.

- 센서 I2C 발견
- Wi-Fi 연결
- receiver IP/port가 예상값
- `send_failures`가 증가하지 않음

실제 센서가 발견되지 않거나 frame counter가 증가하지 않으면 수집을 진행하지 말고 배선·전원·I2C 주소를 확인한다.

### 3. Raspberry Pi에 수집기 복사

PC의 저장소 루트에서 실행한다.

```powershell
scp scripts/thermal_udp_capture.py <pi-user>@<pi-host>:~/safenest-thermal-capture/
scp scripts/validate_thermal_real_capture.py <pi-user>@<pi-host>:~/safenest-thermal-capture/
```

팀원이 이어서 작업할 때의 코드 위치는 다음과 같다.

| 역할 | Git 원본 | 실제 실행/업로드 위치 |
|---|---|---|
| XIAO-ESP32C6 송신기 | `firmware/xiao_esp32c6_thermal90_udp_capture/xiao_esp32c6_thermal90_udp_capture.ino` | Arduino IDE에서 열어 XIAO-ESP32C6에 업로드 |
| Pi 수집기 | `scripts/thermal_udp_capture.py` | `~/safenest-thermal-capture/thermal_udp_capture.py` |
| Pi validator | `scripts/validate_thermal_real_capture.py` | `~/safenest-thermal-capture/validate_thermal_real_capture.py` |

Git 파일 존재만으로 ESP32 업로드 완료를 증명할 수 없다. 업로드 후 Serial Monitor `115200` baud에서 다음을 확인하고 기록한다.

```text
[SafeNest Thermal-90 raw UDP sender]
[Protocol] SafeNest Thermal raw UDP V2: 10080 bytes/frame, 9 chunks, 80 x 62 pixels
[Receiver] <Pi WLAN IP>:5005
```

이전 실제 로그에서는 위 호환 문구가 확인되었지만 `send_failures`가 8회 관찰되었다. 이후 세션에서는 Pi 수집기를 먼저 실행하고 ESP32를 재시작한 뒤 `send_failures`가 더 증가하지 않는지 확인한다.

Pi에서 수신 주소를 확인한다.

```bash
hostname -I
ss -lun | grep ':5005'
mkdir -p ~/thermal-captures
```

### 4. 첫 30초 빈 장면 pilot

Pi 수집기를 먼저 실행하고 XIAO를 켠다.

```bash
python3 ~/safenest-thermal-capture/thermal_udp_capture.py \
  --reassemble-udp-chunks \
  --output ~/thermal-captures \
  --collection-id collection_20260816_pilot01 \
  --subject-id S000 \
  --session-id session_S000_001 \
  --operator-code OP_001 \
  --duration-seconds 30 \
  --source-label EMPTY \
  --sensor-device-id THERMAL90_001 \
  --firmware-version xiao_thermal_udp_v1
```

다음이 생성되는지 확인한다.

```bash
find ~/thermal-captures/collection_20260816_pilot01 -type f | sort
```

조각 재조립 모드에서는 `raw_chunks/`, `raw/`, `decoded_native/`가 함께 생성되어야 한다. `raw/`가 없거나 `.npy`/화면 screenshot만 생성되면 계약형 수집에 실패한 것이다. 그 상태에서 대량 수집·재학습을 시작하지 않는다.

### 5. 수집 직후 Pi에서 validator 실행

```bash
collection=~/thermal-captures/collection_20260816_pilot01
python3 ~/safenest-thermal-capture/validate_thermal_real_capture.py "$collection" \
  --json-out "$collection/validator_result.json"
```

예상되는 초기 제한:

- `PHYSICAL_UNIT_NOT_VERIFIED`
- `EFFECTIVE_FPS_NOT_VERIFIED`
- `TEMPORAL_ORDER_ONLY`
- `CONFIGURED_EFFECTIVE_FPS_DIFFERENCE` (실제 pilot 측정값이 설정 FPS와 다를 때)

이 제한은 실제 수집값 검토 대상이며, 임의로 `VERIFIED`로 바꾸지 않는다.

### 6. 조건별 추가 session 수집

센서 재시작, 설치 변경, subject 변경, 환경 변경마다 새 session ID를 사용한다.

- 빈 장면: `source-label EMPTY`
- 서 있기: `source-label STANDING`
- 앉기: `source-label SITTING`
- 안전하게 눕기: `source-label LYING`
- 거리·각도·부분 가림·배경 변화: metadata 옵션에 기록

LYING을 낙상으로 명명하지 않는다. 실제 전이 event를 수집하려면 별도 안전 승인과 phase annotation 설계가 필요하다.

현재 정적 수집 재시도 대상은 다음과 같다.

- `session_S000_015`: `EMPTY` 재수집
- `session_S000_016`: `STANDING` 재수집
- `session_S000_017`: `LYING` 재수집

재시도 전에는 Pi 수집기를 먼저 실행하고 ESP32를 재시작한다. Serial Monitor의 `send_failures`가 수집 중 증가하지 않는지 확인한다. V2 모드에서 `incomplete_frames`, `checksum_failures`, `conflicting_duplicates`, `invalid_datagrams`가 하나라도 발생하면 대량 수집을 중단하고 Wi-Fi·송신 실패·수신 buffer를 조사한다. 누락 조각을 수동 보정하거나 다음 frame과 연결하지 않는다.

### 7. PC로 수집물 회수

PC의 로컬 PowerShell에서 실행한다.

```powershell
$dest = "$env:USERPROFILE\Documents\rpi_backup\20260816"
New-Item -ItemType Directory -Force -Path $dest
scp -r <pi-user>@<pi-host>:~/thermal-captures/collection_20260816_pilot01 $dest
```

raw capture는 `data/real_capture/` 또는 외부 SSD에 보관하며 Git에 추가하지 않는다.

### 8. PC에서 재검증 및 전달

```powershell
cd C:\Users\KIM TAEGYUN\Documents\ChatGPT\Embedded_SW\safenest-thermal-ai
python scripts\validate_thermal_real_capture.py `
  "$env:USERPROFILE\Documents\rpi_backup\20260816\collection_20260816_pilot01" `
  --json-out "$env:USERPROFILE\Documents\rpi_backup\20260816\validator_result_pc.json"
```

팀에 전달할 것은 collection 전체, Pi/PC validator 결과, 센서·firmware·collector 버전, 실제 FPS, packet loss·decode 오류, UNKNOWN/NOT_VERIFIED 목록이다.

### 9. T-C 종료 판단

- 입력 계약·raw 무결성·시간 정보가 확인되고 도메인 문제가 허용 범위면: 재학습하지 않고 T-C 결과를 문서화한다.
- unit/geometry/시간/설치 조건 불일치 또는 성능 저하가 확인되면: 원인과 증거를 정리해 T-D 재학습 승인을 요청한다.
- T-C 증거와 명시 승인이 없으면 T-D 재학습을 시작하지 않는다.

### 10. T-D 재학습 승인 후 절차

1. 수집물의 권한·동의·checksum을 확인한다.
2. subject/session/event group split을 고정한다.
3. TRAIN 데이터로만 P1 전처리 통계를 fit한다.
4. T-A0~T-A6 validator를 통과시킨다.
5. T-B1~T-B5를 실행하고 validation 기준으로 후보를 고른다.
6. LOCKED_TEST는 마지막 1회만 사용한다.
7. float/INT8 parity와 artifact checksum을 기록한다.

## 현재 미해결 항목

- SNTR UDP V2 송신기와 Pi reassembler는 로컬 Python 회귀 테스트를 통과했고, XIAO ESP32-C6 스케치는 `esp32:esp32 3.3.11` / `esp32:esp32:XIAO_ESP32C6` 대상으로 컴파일을 통과했다(Flash 1,000,948 bytes, RAM 55,952 bytes). 실제 보드 업로드와 Pi 수신은 T-C 통합 시점까지 수행하지 않았다. Git 파일과 실제 보드 binary의 일치 여부는 그때 Arduino IDE 업로드 기록으로 별도 확인해야 한다.
- 실제 Thermal-90 native unit, byte order의 물리적 의미, orientation은 아직 검증하지 않았다.
- `session_S000_004`에서 effective FPS 약 4.3173, sensor counter gap/packet loss 0을 측정했지만, 2초 timing gap 4회 원인은 아직 확인하지 않았다.
- `session_S000_013`은 effective FPS 약 5.7792, frame-counter 오류 0인 유일한 정적 자세 pilot 후보이다.
- `session_S000_011`, `012`, `014`는 재조립 후 frame-counter 오류로 invalid 판정되었다.
- Pi의 실제 WLAN IP와 Wi-Fi 비밀값은 작업자가 입력해야 한다.
- 기존 `Desktop\Thermal_Test\udp_receiver_rpi.py`는 화면 표시·보정 중심의 prototype이며 계약형 수집기로 사용하지 않는다.
- 공개 저장소에는 `wifi_secrets.h`, raw capture, `.tflite` binary를 추가하지 않는다.

## 파일 위치 요약

### 새 저장소의 현재 파일

```text
C:\Users\KIM TAEGYUN\Documents\ChatGPT\Embedded_SW\safenest-thermal-ai\
├── firmware\xiao_esp32c6_thermal90_udp_capture\
│   ├── xiao_esp32c6_thermal90_udp_capture.ino
│   └── wifi_secrets.example.h
├── scripts\thermal_udp_capture.py
├── scripts\validate_thermal_real_capture.py
└── docs\THERMAL90_UDP_CAPTURE_SETUP_KO.md
```

Pi 배포 경로:

```text
~/safenest-thermal-capture/thermal_udp_capture.py
~/safenest-thermal-capture/validate_thermal_real_capture.py
```

### 원본 Desktop prototype

```text
C:\Users\KIMTAEGYUN\Desktop\Thermal_Test\udp_sender_esp32\udp_sender_esp32.ino
C:\Users\KIMTAEGYUN\Desktop\Thermal_Test\udp_receiver_rpi.py
```

원본 prototype과 새 계약형 수집기를 혼동하지 않는다. 실제 수집에는 새 저장소의 `.ino`와 `thermal_udp_capture.py`를 사용한다.

## 완료 체크리스트

- [ ] `wifi_secrets.h`를 로컬에서 생성했고 Git에 나타나지 않는다.
- [ ] XIAO Serial Monitor에서 센서·Wi-Fi·송신 상태를 확인했다.
- [ ] Pi 수집기를 먼저 실행했다.
- [ ] 10,080-byte raw datagram이 `raw/`에 저장됐다.
- [ ] `decoded_native/`, `frames.jsonl`, `annotations.jsonl`, `session.json`, `checksums.sha256`가 생성됐다.
- [ ] Pi와 PC에서 validator를 실행했다.
- [ ] 실제 FPS·loss·unit·orientation 제한을 기록했다.
- [ ] raw capture와 secret을 Git에 추가하지 않았다.
- [ ] T-C 검토와 명시 승인 전에 재학습하지 않았다.
