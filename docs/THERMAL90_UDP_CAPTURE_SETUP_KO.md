# Thermal-90 UDP 수집기 설치·실행 절차

이 문서는 `Thermal_Test`의 10,080-byte 논리 raw frame을 보존하면서, XIAO-ESP32C6와 Raspberry Pi로 **DEVICE_CONTRACT_PILOT**을 수집하는 방법이다. 모델 추론·재학습·낙상 판정은 수행하지 않는다.

## 1. 프로토콜 고정

XIAO는 논리 프레임 하나를 MTU-safe UDP V2 chunk 9개로 전송한다. 각 chunk는 frame/chunk 식별자와 전체 frame CRC32를 가지므로, 손실된 chunk가 다음 frame과 섞이지 않는다.

| 항목 | 값 |
| --- | --- |
| UDP port | 기본 `5005` |
| 논리 frame 크기 | 정확히 `10080` bytes |
| UDP magic/version | `SNTR` / `2` |
| chunk header | 32-byte network byte order |
| 최대 datagram | `1200` bytes |
| chunk payload | 최대 `1168` bytes |
| frame당 chunk | `9` |
| 무결성 | frame ID/index/count/offset/length + 전체 frame CRC32 |
| word 수 | `5040`개의 little-endian `uint16` |
| header | word `0..79` |
| pixel | word `80..5039`, `80×62` |
| header 관찰값 | word `0`: frame counter, `2`: die temp 추정 필드, `5/6`: max/min 추정 필드 |

payload 형식은 기존 `Thermal_Test` 구현과 동일하다. `--reassemble-udp-chunks`는 SNTR V2 header로 frame별 조립하고 CRC32를 확인한다. 구형 1320/1460-byte 조각을 단순 연결하는 방식은 패킷 손실 후 frame 경계를 증명할 수 없으므로 새 수집에 사용하지 않는다. header 의미, 물리 온도 단위, orientation, 실제 FPS는 이번 pilot에서 **확인 대상**이며 확정값이 아니다.

## 2. XIAO-ESP32C6 준비

1. PC의 `firmware/xiao_esp32c6_thermal90_udp_capture/` 폴더를 Arduino IDE로 연다.
2. `wifi_secrets.example.h`를 복사해 같은 폴더에 `wifi_secrets.h`를 만든다.
3. `wifi_secrets.h`에 2.4 GHz Wi-Fi SSID·비밀번호와 Raspberry Pi의 WLAN IPv4 주소를 입력한다. 이 파일은 Git에서 무시되므로 커밋하지 않는다.
4. Arduino IDE에서 XIAO ESP32-C6 보드와 실제 serial port를 선택해 `.ino`를 업로드한다.
5. Serial Monitor `115200` baud에서 다음을 확인한다.
   - `[Protocol] SafeNest Thermal raw UDP V2: 10080 bytes/frame, 9 chunks, 80 x 62 pixels`
   - `[Receiver] <Pi IP>:5005`
   - `send_failures=0`에 가까운 상태

핀과 sensor register 초기화는 `Thermal_Test`의 기존 스케치를 그대로 바탕으로 했다. 배선 변경 전에는 D1/D2/D3/D4/D5/D8/D9/D10 연결을 실제 장비와 대조한다.

## 3. Raspberry Pi 준비

수집기는 외부 Python 패키지가 필요 없다. Pi에서 다음을 실행한다.

```bash
mkdir -p ~/safenest-thermal-capture
```

### 팀원이 확인해야 할 실제 코드 위치

| 역할 | Git 저장소 원본 | Raspberry Pi/Arduino 실제 위치 |
|---|---|---|
| ESP32 송신 코드 | `firmware/xiao_esp32c6_thermal90_udp_capture/xiao_esp32c6_thermal90_udp_capture.ino` | Arduino IDE에서 위 폴더의 `.ino`를 열어 XIAO-ESP32C6에 업로드 |
| Pi 수집기 | `scripts/thermal_udp_capture.py` | `~/safenest-thermal-capture/thermal_udp_capture.py` |
| Pi validator | `scripts/validate_thermal_real_capture.py` | `~/safenest-thermal-capture/validate_thermal_real_capture.py` |

Pi에서 배포된 파일을 확인한다.

```bash
ls -l ~/safenest-thermal-capture/thermal_udp_capture.py \
  ~/safenest-thermal-capture/validate_thermal_real_capture.py
sha256sum ~/safenest-thermal-capture/thermal_udp_capture.py \
  ~/safenest-thermal-capture/validate_thermal_real_capture.py
```

Git의 `.ino` 파일이 존재하는 것과 실제 ESP32에 업로드된 것은 별개다. Arduino Serial Monitor에서 아래 세 줄을 확인해야 현재 펌웨어가 이 수집기와 호환된다고 기록할 수 있다.

```text
[SafeNest Thermal-90 raw UDP sender]
[Protocol] SafeNest Thermal raw UDP V2: 10080 bytes/frame, 9 chunks, 80 x 62 pixels
[Receiver] <Raspberry Pi WLAN IP>:5005
```

`send_failures`가 수집 중 증가하면 업로드 성공 여부와 별개로 해당 세션은 통신 품질 검토 대상으로 보류한다.

PC에서 Pi로 수집기와 validator를 복사한다. `<pi-user>`, `<pi-host>`만 실제 값으로 교체한다.

```powershell
scp scripts/thermal_udp_capture.py <pi-user>@<pi-host>:~/safenest-thermal-capture/
scp scripts/validate_thermal_real_capture.py <pi-user>@<pi-host>:~/safenest-thermal-capture/
```

Pi에서 수신 포트가 사용 중인지 확인한다.

```bash
ss -lun | grep ':5005'
hostname -I
```

`hostname -I`의 Raspberry Pi WLAN IP가 XIAO의 `THERMAL_RECEIVER_IP`와 같아야 한다.

## 4. 30초 빈 장면 시험 수집

**Pi 수집기를 먼저 실행하고, 그 뒤 XIAO를 켠다.** 다음은 가명 subject·operator 예시다.

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

성공하면 다음이 생긴다.

```text
~/thermal-captures/collection_20260816_pilot01/
├── collection.json
└── subjects/S000/sessions/session_S000_001/
    ├── raw/*.udp.bin                 # CRC 통과 후 재조립된 원본 10,080-byte 논리 frame
    ├── raw_chunks/*.bin              # 수신한 32-byte header 포함 UDP datagram
    ├── decoded_native/*_pixels_u16le.bin
    ├── frames.jsonl
    ├── annotations.jsonl
    ├── session.json
    └── checksums.sha256
```

`raw/`에는 resize, crop, rotation, normalization, calibration, colour-map이 전혀 적용되지 않는다. `decoded_native/`도 원본 pixel word를 little-endian `uint16`으로만 분리한 파일이다.

## 5. Pi에서 즉시 검사

```bash
collection=~/thermal-captures/collection_20260816_pilot01
python3 ~/safenest-thermal-capture/validate_thermal_real_capture.py "$collection" \
  --json-out "$collection/validator_result.json"
```

처음에는 `CAPTURE_STRUCTURE_VALID_WITH_LIMITATIONS`, `PHYSICAL_UNIT_NOT_VERIFIED`, `EFFECTIVE_FPS_NOT_VERIFIED`가 나올 수 있다. 이것은 구조 검사 결과이며 T-C·T-D·학습 승인이나 모델 성능 판정이 아니다.

## 6. 조건을 바꿀 때

센서 재시작, 설치 높이/각도 변경, 방/배경 변경, subject 변경은 새 `session_id`로 수집한다. 같은 collection 아래에서 다음처럼 반복한다.

```bash
python3 ~/safenest-thermal-capture/thermal_udp_capture.py \
  --output ~/thermal-captures \
  --collection-id collection_20260816_pilot01 \
  --subject-id S001 \
  --session-id session_S001_002 \
  --operator-code OP_001 \
  --duration-seconds 120 \
  --source-label STANDING \
  --occlusion-condition NONE \
  --background-variation NORMAL_ROOM \
  --sensor-device-id THERMAL90_001 \
  --firmware-version xiao_thermal_udp_v1
```

`LYING`은 안전하게 누운 자세의 source label일 뿐이다. `HUMAN_FALL` 또는 실제 낙상 event로 저장하지 않으며, 보호되지 않은 자유 낙상 실험은 하지 않는다.

## 7. PC로 회수

PC의 로컬 PowerShell에서 collection 폴더를 통째로 회수한다. raw capture는 Git에 추가하지 않는다.

```powershell
$dest = "$env:USERPROFILE\Documents\rpi_backup\20260816"
New-Item -ItemType Directory -Force -Path $dest
scp -r <pi-user>@<pi-host>:~/thermal-captures/collection_20260816_pilot01 $dest
```

PC에서 동일 validator를 다시 실행하고 `validator_result.json`, `collection.json`, session 전체, checksum을 함께 보관한다.

## 문제 해결

| 증상 | 확인 순서 |
| --- | --- |
| Pi가 `0` frame | Pi 수집기를 먼저 실행했는지, XIAO receiver IP/port가 맞는지, 동일 2.4 GHz Wi-Fi인지, Serial Monitor의 Wi-Fi·send failure를 확인 |
| invalid datagram | Serial 문구가 UDP V2인지, magic/version/header/길이가 맞는지 확인. 해당 datagram은 `raw_chunks/`에 보존됨 |
| incomplete frame / CRC 실패 | 대량 수집을 중단하고 Wi-Fi·send failure·chunk loss를 조사. 누락 frame을 다음 frame bytes로 보충하지 않음 |
| checksum 실패 | 수집 종료 후 파일을 수정·이동하지 말고, Pi의 원본 session 폴더에서 다시 회수 |
| validator raw 오류 | `raw/`, `decoded_native/`, `frames.jsonl`을 따로 지우거나 이름 변경하지 말 것 |

이 수집기는 계약형 pilot용이다. 수집물 검토와 명시 승인 없이 재학습, 데이터 승격, 모델 배포, Git push를 수행하지 않는다.
