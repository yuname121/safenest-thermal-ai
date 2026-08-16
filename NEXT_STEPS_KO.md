# SafeNest Thermal AI: 다음 작업 단계 매뉴얼

이 문서는 현재 작업자가 **어떤 순서로 무엇을 해야 하는지** 정하는 실행 기준입니다. 작업 범위는 Thermal-90 온디바이스 AI의 검증·재학습뿐입니다. TCP/UDP, ESP-WROOM-32 펌웨어, 경보/위험도 정책은 여기서 변경하지 않습니다.

## 0. 시작 전 반드시 이해할 사실

- 현재 선택본은 T-B5 오프라인 `FULL_INT8` 후보(318,280 bytes, SHA-256 `fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be`)다. Git에는 이진 파일이 없다.
- 현재 오프라인 근거만으로 Thermal-90 현장 환경의 성능, ESP 배포 가능성, 실제 낙상 검출 성능을 주장할 수 없다.
- `HUMAN_FALL`은 LYING 기반의 **자세 proxy**다. 실제 낙상 event 라벨이 아니다.
- REAL_EVAL_DEVELOPMENT 결과는 float Macro F1 `0.5939`, INT8 Macro F1 `0.6390`이었다. 이 데이터에는 엄격히 분리된 pristine LOCKED_TEST가 없고 near-duplicate 한계가 있으므로, 이것을 최종 성능 수치로 사용하지 않는다.
- 구형 `v0.1.0` 모델과 min-max interpreter는 의도적으로 포함하지 않았다. 새 B단계 전처리와 섞지 않는다.

**지금의 단계는 T-C(실장치/도메인 검증) 준비 및 수행이다.** T-D 재학습은 T-C의 증거를 검토하고 사람이 승인한 뒤에만 시작한다.

## 1. 내 로컬 환경을 준비한다

### 1-1. 저장소와 브랜치를 확인한다

```powershell
git clone https://github.com/rla1729/safenest-thermal-ai.git
cd safenest-thermal-ai
git switch -c experiment/thermal-tc-pilot
```

이미 clone되어 있으면 `git status`가 깨끗한지 확인하고, 작업마다 `experiment/thermal-...` 브랜치를 새로 만든다. `main`에 직접 실험 코드를 넣지 않는다.

### 1-2. Python 가상환경을 만든다

Python 3.10 또는 3.11을 권장한다. PowerShell에서 다음을 실행한다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

macOS에서 전체 학습을 재현할 때는 `requirements-mac.txt`를 사용한다. 설치 문제를 피하려고 기존 시스템 Python이나 전역 패키지에 섞어 설치하지 않는다.

### 1-3. 기준선 검증을 먼저 실행한다

새 기능을 만들기 전에 다음을 실행해 현재 복사본의 계약/manifest가 깨지지 않았는지 확인한다.

```powershell
python -m pytest tests/test_thermal_t_a0.py tests/test_thermal_t_a1.py tests/test_thermal_t_a2.py tests/test_thermal_t_a3.py tests/test_thermal_t_a4.py tests/test_thermal_t_a5.py tests/test_thermal_t_a6.py tests/test_thermal_t_a6_stage2.py tests/test_thermal_t_b0.py tests/test_thermal_t_b1.py tests/test_thermal_t_b2.py tests/test_thermal_t_b3.py tests/test_thermal_t_b4.py tests/test_thermal_t_b5.py tests/test_thermal_real_capture_validator.py -q
```

결과와 사용한 Python/패키지 버전을 작업 노트에 남긴다. TensorFlow 또는 외부 artifact 부재로 실패한 테스트는 실패 원문을 보존하고, 임의로 expectation이나 수치를 고치지 않는다.

## 2. B단계 artifact를 수령하고 동일성을 확인한다

**담당자가 승인된 외부 저장소 위치를 제공할 때만** 선택된 `.tflite`를 수령한다. 파일을 `artifacts/` 또는 승인된 SSD에 두고 Git에 추가하지 않는다.

```powershell
Get-FileHash -Algorithm SHA256 .\artifacts\thermal_full_int8.tflite
(Get-Item .\artifacts\thermal_full_int8.tflite).Length
```

두 값이 각각 위 SHA-256과 `318280`인지 확인한다. 하나라도 다르면 그 파일을 사용하지 말고 수령 담당자에게 재확인을 요청한다. 확인 결과(경로는 필요시 비공개 처리), 해시, 크기, 수령 날짜를 비식별 작업 기록에 남긴다.

## 3. T-C 파일럿 수집을 설계한다 — 아직 재학습하지 않는다

먼저 아래 문서를 읽는다.

1. `docs/20260814_Codex_Thermal_Real_Data_Acquisition_Guide_KO_01.md`
2. `docs/20260814_Codex_Thermal_Real_Data_Acquisition_Contract_EN_01.md`
3. `docs/20260815_Codex_Thermal_Runtime_Temporal_Handoff_KO_01.md`
4. `docs/THERMAL90_UDP_CAPTURE_SETUP_KO.md` — XIAO-ESP32C6와 Raspberry Pi를 쓸 때의 실제 raw-capture 설치·실행 절차

현재 수집 구현은 `firmware/xiao_esp32c6_thermal90_udp_capture/`와 `scripts/thermal_udp_capture.py`다. 이 구현은 `Thermal_Test`의 10,080-byte little-endian 논리 raw frame을 보존하고, `SNTR` UDP V2의 frame ID/chunk index/offset/length/CRC32로 MTU-safe하게 전송·재조립한다. 화면 표시·정규화·모델 추론 대신 원본 datagram, 재조립 raw frame, native pixel, provenance, checksum을 남긴다. 구형 blind stream 재조립은 기존 오류 증거를 읽기 위한 진단 호환 모드일 뿐 새 수집에 사용하지 않는다.

수집 세션마다 다음을 지킨다.

- 원시 packet/프레임, native decode 결과, 프레임별 provenance(JSONL), 세션 메타데이터, checksum을 함께 보존한다.
- 입력 shape, dtype, 단위, byte order, 90° orientation, 좌표계, timestamp/sequence continuity를 기록한다.
- 장치 위치·시야·거리·배경·가림·움직임 조건을 계획하고, 동일 인물/세션/소스가 나중에 train과 test에 섞이지 않게 식별자를 관리한다.
- 참가자 동의, 보관 기간, 접근 권한, 익명화 기준을 실제 수집 전에 책임자와 확인한다.
- 원시 데이터는 `data/real_capture/` 또는 승인된 외부 저장소에만 둔다. Git에는 올리지 않는다.

수집 직후 구조 검사를 실행한다.

```powershell
python scripts/validate_thermal_real_capture.py --help
python -m pytest tests/test_thermal_real_capture_validator.py -q
```

`--help`로 현재 validator의 필수 인자와 directory contract를 확인한 다음 실제 세션 경로를 지정해 검증한다. validator 통과는 파일 계약 통과일 뿐, 모델 성능이나 재학습 승인이 아니다.

## 4. T-C 도메인 검증을 수행한다

각 파일럿 세션에서 다음 검증 표를 작성한다. 부적합이면 **수집/decoder/전처리 문제를 먼저 해결**하고 데이터나 기준 모델을 섣불리 바꾸지 않는다.

| 확인 대상 | 합격 기준 | 불합격 시 조치 |
| --- | --- | --- |
| 입력 계약 | `[62,80,1]`, dtype/단위/범위/방향이 문서와 일치 | raw→canonical 변환과 provenance를 수정·재검증 |
| 시간 연속성 | 누락/중복/역전 없이 session 순서가 증명됨 | packet/decoder 로그부터 보존하고 원인 분리 |
| 라벨 의미 | 실제 event와 자세 proxy를 구분해 기록 | `HUMAN_FALL`을 실제 낙상으로 재명명하지 않음 |
| 도메인 차이 | 설치 조건별 오류·불확실성·대표성 수치화 | 추가 수집 계획 수립, TRAIN/TEST 혼합 금지 |
| artifact 동일성 | 확인된 B5 SHA/크기와 일치 | artifact 재수령·재확인 |

평가에는 손대지 않은 holdout 세션/그룹을 남긴다. 기존 development 평가와 새 현장 자료를 한 표에 섞어 최종 성능처럼 보고하지 않는다.

### T-C 종료 결정

- 도메인 차이가 허용 범위이고 계약 위반이 없다면: 재학습하지 않는다. T-C 결과와 한계를 문서화하고 다음 의사결정을 요청한다.
- 계약 위반·대표성 부족·성능 저하가 확인되면: 원인, 근거 파일의 checksum, 영향 범위, 추가 수집량을 정리하여 **T-D 재학습 승인**을 요청한다.

## 5. T-D 재학습은 승인 후에만 한다

승인 전에는 이 절을 실행하지 않는다. 승인받은 뒤에도 아래 순서를 고정한다.

1. 새 데이터를 manifest에 등록하고 원본, 동의/사용권한, 수집조건, checksum을 확인한다.
2. source/session/person 단위 group split을 먼저 고정한다. `LOCKED_TEST`는 끝까지 건드리지 않는다.
3. 전처리 통계(예: global z-score)는 **TRAIN만으로 fit**하고 validation/test에는 transform만 적용한다.
4. T-A0~T-A6 validator를 실행해 source identity, raw unit, geometry, temporal policy, label semantics, split을 고정한다.
5. T-B0~T-B5 실험을 같은 계약으로 실행한다. 모델 선택은 validation만 사용하고, 후보/seed/환경/manifest/hash를 기록한다.
6. 최종 후보를 한 번만 LOCKED_TEST로 평가한다. 결과가 나빠도 test에 맞춰 반복 튜닝하지 않는다.

실행 진입점은 `scripts/run_thermal_t_b1.py`부터 `scripts/run_thermal_t_b5.py`까지다. 각 스크립트의 `--help`와 해당 `docs/reports/` 문서를 읽은 뒤 명령을 확정한다. 이전의 `thermal_prep.py`, `thermal_train.py` 방식처럼 source split을 합치거나 frame random split을 쓰면 안 된다.

## 6. 정량화와 후보 고정

새 모델 후보는 float와 FULL_INT8이 동일한 canonical 입력·클래스 순서·전처리 계약을 사용함을 증명해야 한다. T-B4 equivalence와 T-B5 robustness/latency 검증을 다시 통과시킨 뒤 다음을 기록한다.

- `.tflite` SHA-256, byte size, input/output tensor shape·dtype·quantization 값
- class map과 전처리 parameter의 출처(TRAIN manifest hash)
- float/INT8 차이, per-class 지표, seed 분산, 실패 사례
- 평가 데이터의 split identity와 현장 조건

이진 artifact는 Git에 올리지 않는다. 외부 artifact 저장소의 경로/권한은 담당자에게만 공유하고, Git에는 checksum과 재현 가능한 manifest만 남긴다.

## 7. Codex와 Git으로 안전하게 협업한다

Codex에 요청할 때는 예를 들어 다음처럼 범위와 승인 여부를 분명히 쓴다.

> `experiment/thermal-tc-pilot`에서 real-capture validator만 개선해줘. raw 데이터와 artifact는 건드리지 말고, 변경 후 테스트만 실행해줘. 커밋·push·PR은 하지 마.

수정 검토가 끝난 뒤에만 다음을 수행한다.

```powershell
git status
git diff --check
git add <검토한 파일만>
git commit -m "docs: record thermal T-C pilot contract"
git push -u origin experiment/thermal-tc-pilot
```

push나 PR은 외부 공개/공유 변경이므로 명시적으로 승인된 경우에만 한다. 특히 `main` 병합 전에는 테스트 결과, 데이터 권한, artifact checksum, 과장 없는 성능 표현을 다시 검토한다.

## 작업 완료 체크리스트

- [ ] 기준선 테스트/validator 결과를 남겼다.
- [ ] B5 artifact SHA와 크기를 확인했거나, 아직 수령하지 못했음을 명시했다.
- [ ] 실제 수집의 동의·보관·접근 권한을 확인했다.
- [ ] raw/native/provenance/checksum을 모두 보존했다.
- [ ] 현장 holdout 그룹을 train과 섞지 않았다.
- [ ] `HUMAN_FALL`을 실제 낙상으로 과장하지 않았다.
- [ ] 재학습은 T-C 근거와 명시 승인 뒤에만 시작했다.
- [ ] Git에 raw data·model binary·secret을 올리지 않았다.
- [ ] 커밋/push/PR은 검토와 명시 승인 뒤에만 실행했다.
