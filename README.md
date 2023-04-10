# Univariate Monitoring

## classify_fault
classify_fault는 단변량 통계를 이용하여 변수 유형을 확인하고, 각 유형의 장애를 감지하는 패키지입니다.

---

### 패키지 구성
```
arduino
Copy code
classify_fault/
    utils/
        convert_type.py
        __init__.py
    bias_monitoring.py
    check_boundary.py
    check_drift.py
    check_frozen.py
    check_dynamics.py
    set_config.py
    update_avg_std.py
    fault_detection.py
    __init__.py
    test/
        __init__.py
        test.py
    config/
        variable_confign.yaml
    example.ipynb
```
---

### 모듈 설명
    set_config.py
    데이터 분석에 필요한 변수 설정을 위한 모듈입니다.

    fault_detection.py
    변수 유형을 확인하고, 각 유형의 장애를 감지하는 모듈입니다.

    bias_monitoring.py
    각 변수의 Bias를 모니터링하기 위한 모듈입니다.

    check_boundary.py
    Boundary(경계)를 모니터링하기 위한 모듈입니다.

    check_drift.py
    Drift(드리프트)를 모니터링하기 위한 모듈입니다.

    check_frozen.py
    Frozen(고정)을 모니터링하기 위한 모듈입니다.

    check_dynamics.py
    Dynamics(동적)를 모니터링하기 위한 모듈입니다.

    update_avg_std.py
    평균과 표준편차를 업데이트하기 위한 모듈입니다.

    convert_type.py
    데이터 타입 변환을 위한 모듈입니다.

---

### 테스트
classify_fault 패키지를 테스트하기 위해서는 test 디렉토리의 test.py를 실행해주세요.
```
python test/test.py
```
---

### 예시
```
import numpy as np
from classify_fault.fault_detection import detect_fault

data = np.array([1, 2, 2, 2, 10, 10])

type_to_check = {
    "frozen": True,
    "boundary": True,
    "dynamics": True,
    "drift": True
}

frozen_threshold = 0.5
tracking_size = 3
boundary_limits = (12, 0)
dynamic_threshold = 2.5
drift_params = {"average": 2, "cusum_threshold": 4.0, "ewma_alpha": 0.2}

fault_detected = detect_fault(data, tracking_size, type_to_check, frozen_threshold, boundary_limits, dynamic_threshold, drift_params)

print(f"Is fault detected? {fault_detected['fault_detected']}")
# Is fault detected? False
```