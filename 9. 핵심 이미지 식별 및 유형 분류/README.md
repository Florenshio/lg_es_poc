# 1. 목적 및 이슈

### 목적
    1. 문서 안의 중요한 도면 정보 감지
    2. 감지된 도면의 유형 분류
### 이슈
    - 도메인 지식 부족으로 인한 클래스 라벨링의 어려움
    - LG 폰트 적용 불가로 인한 문서 형태의 깨짐
        - 표의 선이나 글자가 도면 이미지와 섞이는 경우 발생
    - `비정형문서 내 핵심 영역 식별` task와 기능이 통합된 YOLOv8 모델 학습

# 2. 산출물 및 결론
- 산출물
    1. 1차 분류 모델 (YOLOv8 Detection Model)
    2. 2차 분류 모델 (YOLOv8 Classification Model)
- 결론
    - 이미지 Detection 모델로 1차 분류 후 이미지 Classification 모델로 2차 분류
    - mAP50 : 94%
    - Top1 accuracy : 94%
    - 클래스 별 데이터 수 불균형으로 인한 성능 저하 발견


# 3. 프로세스
![이미지 감지분류_프로세스](./png/이미지%20감지분류_프로세스.png)


> Step 1. 1차 분류 모델 학습

1. 문서 내의 핵심 정보에 해당하는 각 이미지에 대해 바운딩 박스 및 클래스 라벨링
2. 이미지와 라벨링 데이터를 YOLOv8에 Input
    - 라벨링 데이터 : 바운딩 박스의 좌표와 클래스가 기록된 txt파일
3. YOLOv8 Detection Model 학습

> Step 2. 2차 분류 모델 학습

1. 1차 분류 모델을 활용해 문서 내에 존재하는 도면 이미지를 Crop
2. Crop된 이미지에 알맞는 클래스 라벨링
    - 학습 데이터 디렉토리 안에 각 클래스 별 디렉토리 생성 후 알맞는 디렉토리에 이미지를 넣는 방식
3. YOLOv8 Classification Model 학습

> Step 3. 핵심 이미지 식별 및 유형 분류

1. 입력된 문서를 PDF 변환 후 각 페이지 별로 이미지 추출
2. YOLOv8 Detection Model로 문서 내 중요한 도면 식별 및 1차 분류
    (1) 1차 분류 클래스 3가지
        1. Drawing (도면)
        2. Table (표)
        3. Column (열)
            - Table과 Column은 핵심 Text 식별 task에 필요한 클래스
            - 이미지 식별/분류 task는 Drawing 결과 값만 필요
3. YOLOv8 Classification Model로 Crop된 도면 이미지 2차 분류
    (1) 2차 분류 클래스 5가지
        1. Cell
        2. Assy
        3. Roll
        4. Rode
        5. Etc

# 4. 모델

### YOLOv8 Detection Model
- 이미지 1차 분류 모델
- 이미지 식별과 분류를 동시에 수행
- `비정형문서 내 핵심 영역 식별` task와 기능이 통합됨
- 추론 결과 값:
    1. 이미지 바운딩 박스 좌표
    2. 유형 분류 결과
    3. confidence 값
    
- 클래스 3가지:
    1. Drawing (도면) → 이미지 식별/분류 task 용
    2. Table (표) → 핵심 TEXT 식별 task 용
    3. Column (열) → 핵심 TEXT 식별 task 용
    => 해당 task에서는 `Drawing`에 해당하는 이미지만 Crop하여 2차 분류
    
- 학습 시간 : 약 16 시간 소요
- 학습 하이퍼 파라미터:
    - epochs = 1000
    - imgsz=975
    - patience = 50
    - save_period = 100
    - batch = 4
    - scale=0.3
- 최고 성능 epoch 150
    
- metric:   
    - train/box_loss = 0.43154
    - train/cls_loss = 0.34812
    - train/df1_loss = 0.95908
    - mAP50 = 0.94254
    - mAP50-95 = 0.83685
    - val/box_loss = 0.49949
    - val/cls_loss = 0.41211
    - val/df1_loss = 0.9564

### YOLOv8 Classification Model
- 이미지 2차 분류 모델
- 1차 분류의 `Drawing`에 해당하는 이미지 유형 분류

- 추론 결과 값:
    1. 유형 분류 결과
    2. confidence 값

- 클래스 5가지:
    1. Cell
    2. Assy
    3. Roll
    4. Rode
    5. Etc

- 학습 시간 : 약 1 ~ 2 시간 소요
- 학습 하이퍼 파라미터:
    - epochs=1000
    - imgsz=448
    - patience =50
    - save_period = 50
    - batch = 4
    - scale=0.3
- 최고 성능 epoch 66

- metric:
    - train/loss = 0.002
    - accuracy_top1 = 0.94531
    - val/loss = 0.1209

# 5. 개발 환경

- AWS
- Ubuntu 18.04.5 LTS
- Jupyter notebook
- T4 GPU 4개

# 6. 개발 사항

### 데이터
- 데이터 출처
    - LG 에너지 솔루션 자동차부문 문서 약 2만 3천건(Control Plan, Gate Review, BOM, PFMEA, DFMEA, 제품사양서, 도면)
    - Gate Review, 제품사양서, 도면으로 한정
- 데이터 포맷 및 예시
    - 도면 문서만 존재하는 pptx 파일 → pdf 변환 → 각 페이지 별로 png 변환
    - 일반 문서 안에 도면이 삽입되어있는 pptx 파일 → pdf 변환 → 각 페이지 별로 png 변환
- 전처리 사항
    - 1차 분류 모델 : 1024x1024 크기로 이미지 resize
    - 2차 분류 모델 : 460x460 크기로 이미지 resize

### 검증 사항
- 1차 분류에서 Drawing(도면)으로 분류된 이미지만 crop하여 2차 분류 하도록 되어있는가

### 주요 라이브러리
- ultralytics
- pandas
- numpy