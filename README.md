# lg_es_poc

![LG_PoC 전체 프로세스](./png/LG_POC%20전체%20프로세스.png)

# 프로젝트 정보

### 1. 기간
2023.09 ~ 2023.12

### 2. 인원 및 역할

| **task**              | **담당**        | 핵심 기술 등급 (A, B, C) | 적용한 모델 (패키지)                                                              |
| --------------------- | ------------- | ------------------ | ------------------------------------------------------------------------- |
| 01. 문서 lake 구축        | 장지연 백광현 → 황영준 | -                  | 1. AWS  <br>2. 사이냅 문서 필터  <br>3. Open Search                              |
| 02. 키워드를 이용한 문서 검색    | 조영래 황영준       | C                  | Opensearch SQL                                                            |
| 03. 문서 카테고리 별 분류      | 황영준           | A                  | 1. w2v  <br>2. doc2vec  <br>3. Deep Neural Network                        |
| 04. 문서 분포 분석          | 조영래           | C                  | 1. TSNE 차원 축소  <br>2. Cluster Analysis                                    |
| 05. 유사 문서 검색          | 조영래           | C                  | 1. cosine_similarity  <br>2. Opensearch SQL                               |
| 06. 키워드 분석       | 조영래           | C                  | 1. TF-IDF  <br>2. WordCloud                                               |
| 07. 정형문서 내 핵심 영역 식별   | 장지연 백광현 → 황영준 | A                  | 1. openpyxl  <br>2. PaddleOCR                                             |
| 08. AI OCR            | 백광현 → 박민지     | B                  | PaddleOCR                                                                 |
| 09. 핵심 이미지 식별 및 유형 분류 | 조영래 김홍준       | A                  | YOLOv8                                                                    |
| 10. 비정형문서 내 핵심 영역 식별  | 박민지 김홍준       | A                  | 1. YOLOv8  <br>2. PaddleOCR  <br>3. cosine similarity  <br>4. rouge score |

### 3. 프로젝트 목표

- 내용 기반의 문서 유형 자동 분류
    - 문서의 제목 혹은 특정 키워드를 기반으로 분류하는 것이 아니라, 실제로 문서의 내용을 바탕으로 분류하는 모델을 목표로 함
- 문서 내의 핵심 영역 식별 및 분류
    - 핵심 영역 : 문서 내에서 핵심 정보가 존재하는 영역
    - LG 에너지솔루션의 데이터를 분석한 결과 해당 PoC에서는 핵심 영역이 표에만 해당되는 것으로 정의함

### 4. 대상 및 범위

LG 에너지 솔루션 자동차 부문에서 유통되는 문서 및 도면

1. 자동차 부문 전체 문서 : 74만 건
2. 대상 확장자 확정(ppt, pptx, xls, xlsx, xlsm, doc, docx, pdf, msg, dwg, jpg, png, txt) : 74만 건 => 65만 건
3. 주요 문서 7종 추출(Poc 과정에서는 7가지 유형으로 한정) : 65만 건 => 72,082 건
    - 문서 유형 자동 분류 Category:
        - Control Plan (공정 프로세스)
        - BOM (자재 명세서)
        - PFMEA
        - DFMEA
        - Gate Review
        - 제품사양서
        - 도면
4. 영문과 한글 이외 문자 제거, 중복파일(최종 rev만 남김) 제거 : 72,082 건 => 28,058 건
5. DRM적용 파일 및 오류 파일 제거(4,669 건) : 28,058 건 => 23,489 건

- 문서유형
    - MS office (xls, xlsx, xlsm, ppt, pptx, doc, docx, msg)
    - PDF
    - txt
    - image file (png, jpg)
    - dwg → dxf → png

# 핵심 기능

- 문서 카테고리 별 분류
- 정형문서의 핵심 영역 식별
- 비정형문서의 핵심 영역 식별
- 핵심 이미지 식별 및 유형분류

# 주요 Task 설명

### 3. 문서 카테고리 별 분류

1. 문서 Lake의 모든 문서에 대한 plain text를 받아온다.
2. 모든 text를 전처리하고 명사만 추출한다.
3. 명사 추출된 모든 문서의 text를 Word2Vec 방식으로 임베딩하여 최종적으로 문서 임베딩 벡터를 만든다.
4. 문서 임베딩 벡터를 통해 Fully Connected Layer 분류 모델을 학습한다.
5. input 값으로 문서 임베딩 벡터를 받으면 분류 모델을 통해 문서 카테고리 분류를 한다.
6. 생성된 모든 문서에 대한 문서 임베딩 벡터는 `4. 문서 Context 기반 분포도` , `5. 내용 기반 유사 문서 검색` , `10. 비정형 문서의 핵심 영역 식별` 태스크에 사용된다.
7. 명사 추출된 모든 문서의 text는 `6. 키워드 분석` 태스크에 사용된다.
8. 분류 모델을 통해 나온 문서 카테고리 추론 결과 값은 `4. 문서 Context 기반 분포도` 태스크에 사용된다.

### 4. 문서 분포 분석

1. input 값으로 문서 임베딩 벡터와 문서 카테고리 분류 추론 결과 값을 받는다.
2. 고차원의 문서 임베딩 벡터를 t-SNE 방식으로 2차원 변환한다.
3. 카테고리 별 문서들의 2차원 임베딩 벡터를 산점도로 표현한다.

### 6. 키워드 분석
    
1. `3. 문서 카테고리 별 분류` 에서 받은 모든 문서에 대한 명사 추출된 text를 TF-IDF 벡터로 만든다.
2. 각 문서 별로 TF-IDF 값이 높은 순으로 정렬된 키워드 데이터를 얻는다.
3. 특정 문서의 핵심 키워드 리스트와 워드 클라우드를 출력한다.

### 7. 정형문서의 핵심 영역 식별

- _**방법 1 (스캔된 정형 문서(PNG)에서 핵심 영역 식별)**_

1. input 값으로 원본 excel 파일(정형 문서)의 각 sheet들이 PNG로 변환된(스캔된) 파일을 받는다.
2. opencv 라이브러리를 사용하여 셀 구분선을 인식한 후, 셀 구조를 파악한다.
3. 각 셀 별로 OCR를 수행한다.
4. 핵심 정보 key에 해당하는 셀을 탐하고, key에 해당하는 value 셀을 찾고 유효성을 검사한다.
5. 핵심 영역이 확정되면, criticlity에 따라 색으로 구분하여 영역을 표시한다.

- _**방법 2 (정형 문서 파일에서 핵심 영역 식별)**_

1. input 값으로 원본 excel 파일(정형 문서)을 받는다,
2. openpyxl 라이브러리를 사용하여 excel 안에 있는 모든 데이터를 읽어온다.
3. 핵심 정보 key에 해당하는 셀을 탐색하고, key에 해당하는 value 셀들을 찾고 유효성 검사를 진행한다.
4. 핵심 영역이 확정되면, criticlity에 따라 색으로 구분하여 영역을 표시한다.
5. 최종 결과로 해당하는 핵심 영역이 표시된 excel 파일이 리턴된다.

### 8. OCR

1. input 값으로 PNG 파일로 변환된 페이지 혹은 YOLOv8 detection 결과인 crop image(PNG)를 받는다.
2. paddleOCR을 사용하여 PNG 파일에 있는 텍스트를 추출한다.
3. 추출 결과는 `7. 문서 영역식별` `9. 이미지 식별/분류` `10. 핵심 Text 식별` 태스크에 사용된다.

### 9. 핵심 이미지 식별 및 유형분류

1. 추출된 이미지 파일(PNG)을 통해 YOLOv8 Detection Model, YOLOv8 Classification Model을 학습한다.
2. input 값으로 PNG 파일을 받으면 YOLOv8 Detection Model을 통해 핵심 이미지 식별 및 1차 유형 분류를 한다.
3. 핵심 도면 이미지로 나온 결과 값만 추출해 YOLOv8 Classification Model로 2차 유형 분류를 한다.

### 10. 비정형문서의 핵심 영역 식별

1. input 값으로 OCR 결과 추출된 텍스트들을 받는다.
2. `3. 문서 자동 분류` 태스크와 동일한 전처리 방식으로 텍스트를 전처리하고, 명사를 추출하여 w2v 임베딩으로 임베딩 벡터를 만든다.
3. 미리 정의되어 있는 핵심 text의 임베딩 벡터와 추출된 임베딩 벡터를 코사인 유사도 비교하여 threshold 0.8이상인 값들만 rouge score를 적용한다.
4. 추출된 text에서 rouge score를 적용하여 threshold 0.4이상인 값들을 핵심 text라고 식별한다. (코사인 유사도와 rouge score의 threshold를 둘 다 만족해야함.)