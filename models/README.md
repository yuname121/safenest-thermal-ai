# Model artifact policy

현재 선택 기준선은 T-B5의 `FULL_INT8` 후보입니다.

- SHA-256: `fa9730c29535477a3994c11e664474a0ca0116afaaa172889f47446ab2ac46be`
- 파일 크기: 318,280 bytes
- 입력/출력: `[1, 62, 80, 1]` → `[1, 3]`
- 클래스 순서: `NOT_HUMAN`, `HUMAN_NORMAL`, `HUMAN_FALL`

이 모델은 외부 artifact 저장소에 있으며 Git에는 저장하지 않습니다. 수령 후 SHA-256과 파일 크기를 확인하고, T-C 데이터 계약 검증을 통과한 환경에서만 평가에 사용합니다. 이 저장소에는 과거 `v0.1.0` 모델을 의도적으로 포함하지 않았습니다.
