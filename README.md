# SafeNest Thermal AI

Thermal-90 센서 기반 온디바이스 열화상 AI의 **검증과 재학습**을 위한 독립 작업공간입니다. 대상 하드웨어는 ESP-WROOM-32이지만, 이 저장소의 범위는 데이터·모델·정량화 검증이며 TCP/UDP, 펌웨어 통합, 경보/위험도 판단은 다루지 않습니다.

## 현재 기준선

- T-A0~T-B5 오프라인 단계의 코드와 검증 근거(manifest)를 보존했습니다.
- 다음 단계는 T-C: 실제 Thermal-90 장치/설치 환경에서 수집한 데이터로 입력 계약과 도메인 차이를 검증하는 일입니다.
- 선택된 오프라인 후보는 `FULL_INT8`, SHA-256 `fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be`, 318,280 bytes입니다. 이 이진 파일은 Git에 포함하지 않으며 승인된 외부 artifact 저장소에서만 수령·검증합니다.
- 클래스 `HUMAN_FALL`은 현재 데이터셋에서 **누운 자세(LYING) 유래의 자세 proxy**입니다. 실제 낙상 사건을 검출·보장한다는 주장이 아닙니다.

## 시작 위치

1. 작업 전 [NEXT_STEPS_KO.md](NEXT_STEPS_KO.md)를 처음부터 읽습니다.
2. 보존된 인수인계 문서는 [docs/20260815_Codex_Thermal_Runtime_Temporal_Handoff_KO_01.md](docs/20260815_Codex_Thermal_Runtime_Temporal_Handoff_KO_01.md)입니다.
3. 실제 데이터 수집 기준은 [docs/20260814_Codex_Thermal_Real_Data_Acquisition_Guide_KO_01.md](docs/20260814_Codex_Thermal_Real_Data_Acquisition_Guide_KO_01.md)와 `scripts/validate_thermal_real_capture.py`입니다.
4. XIAO-ESP32C6 → Raspberry Pi UDP raw-capture 구현과 실행 절차는 [docs/THERMAL90_UDP_CAPTURE_SETUP_KO.md](docs/THERMAL90_UDP_CAPTURE_SETUP_KO.md)입니다.
5. 현재 상태와 다음 담당자 작업 순서는 [docs/20260816_Thermal_OnDevice_AI_Handoff_KO.md](docs/20260816_Thermal_OnDevice_AI_Handoff_KO.md)입니다.

## 디렉터리

| 경로 | 용도 |
| --- | --- |
| `datasets/thermal/` | 표준화·분할·학습/평가 파이프라인 및 T-A/T-B 검증 manifest |
| `scripts/` | T-A0~T-B5 생성·검증과 실제 수집 계약 validator |
| `firmware/xiao_esp32c6_thermal90_udp_capture/` | Thermal_Test 호환 10,080-byte 논리 raw frame을 SNTR UDP V2 chunk로 보내는 스케치. Wi-Fi 비밀값은 로컬 `wifi_secrets.h`에만 둠 |
| `tests/` | 단계별 회귀 테스트 |
| `docs/` | 인수인계·수집 계약·오프라인 실험 보고서 |
| `data/`, `artifacts/` | 로컬/승인된 외부 저장소에서만 관리할 비추적 대용량 데이터와 모델 |

## 제한

원본 팀 작업물의 파일별 저작권·공개 권한을 확인하기 전에는 외부 데이터, 새 모델 바이너리, 개인 식별 가능 수집물을 추가하거나 배포하지 마세요. 이 저장소는 원본 공개 작업물에서 선별한 작업용 파생본이며, 출처는 [docs/PROVENANCE.md](docs/PROVENANCE.md)에 기록되어 있습니다.
