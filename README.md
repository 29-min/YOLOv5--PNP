# yolov5--PNP
yolov5를 활용한 교내 이동량 분석
> https://github.com/ultralytics/yolov5
## 1. 주제 선정 및 사유

### 주제: 교내 혼잡도 통계를 작성하고 이를 기반으로 혼잡도 예상 시스템을 구축한다.

### 사유

- 교내 공간 활용 최적화
    - 캠퍼스 내에서 혼잡한 시간과 공간을 실시간으로 모니터링 하면 학생과 교직원이 효율적으로 공간을 활용할 수 있을 것이다. 특히 우리 학교 교내식당의 경우 특정 시간대에 사람이 몰리는 경향이 있는데, 요일별 / 시간대별 통계를 작성하고 이를 공개한다면 인구가 분산되어 사람들이 시간을 더욱 효율적으로 사용할 수 있을 것이다.
- 데이터 기반 문제 해결
    - AI 기반으로 데이터를 축적하고 이를 통해 캠퍼스 운영의 문제점을 정량적이고 객관적으로 분석하고, 시간대별 혼잡도를 기준으로 운영상의 개선 방안을 제시할 수 있다.
- 기술적 난이도와 실현 가능성
    - PNP AI 스터디 팀이 코드를 구현하고 이를 운용하기에 적절한 난이도를 가진다. 사전 학습된 YOLO 모델을 활용해 Object Detection을 구현하고 이를 응용하여 통계작성을 하여 결과물을 만들었다.

## 2. 목표

YOLO (You Only Look Once) 모델을 활용하여 특정 공간의 혼잡도를 분석하기 위해 객체 탐지(Object Detection)를 구현하고, 탐지된 객체 수를 카운팅하여 시간별로 통계 데이터를 생성하고 이를 응용한다.

## (기본 개념)
- - -
### a. 영상 처리: 
  디지털 이미지나 비디오에서 유용한 정보를 추출하고 분석하는 것으로,
  CV의 한 분야이다. 이미지, 필터링, 객체 인식, 추적, 분할 등 다양한 작업을 포함함.

![Object Dection, Localization, Classification](https://github.com/29-min/yolov5--PNP/blob/main/Detection%2C%20Localization%2C%20Classification.png)
### b. Object Dection: 
  이미지나 비디오 내에서 특정 객체를 식별하고 대항 객체의 위치를 지정하는 기술로
  단순한 분류(Classification)와 달리, 객체의 존재 여부와 함께 위치 정보(bounding box)를 제공하여 정밀 분석이 가능하다.

### c. Localization: 
  이미지나 비디오 내에서 객체의 위치를 찾는 과정으로, 객체의 경계를 나타내는 bounding box를 생성하고 객체의 크기와 위치를 파악하여 정보를 제공한다.

### d. Classification: 
  이미지나 비디오 내의 객체를 특정 카테고리로 분류하는 작업. 이미지 내의 객체가 사람, 자동차, 고양이 등 어떤 클래스에 속하는지 식별한다. 

### e. YOLO(You Only Look Once): 
  객체 인식을 실시간으로 수행할 수 있는 딥러닝 모델로 이미지를 X * X 그리드로 분할, 각 그리드 셀이 bounding box와 클래스의 확률을 예측함. 이 과정은 단일 신경망을 통해 한 번에 이루어지므로 속도가 빠르다는 이점이 있다. 

#### f. R-NN 대비 YOLO가 갖는 이점:
  R-CNN(Regin-based Convolutional Nerual Networks)은 객체 검출에서 높은 정확도를 제공하지만, 여러 단계로 이뤄진 층으로 구조가 복잡하고 속도가 느리다. YOLO는 단일 신경망이며, 전체 이미지를 고려하여 예측하기 때문에 전반적인 컨텍스트를 반영한 정확한 검출이 가능하다.


## 3. 주요 코드 설명

### **YOLO 적용 코드의 주요 기능:**

1. **프레임 선택:**30프레임마다(1초 간격) 프레임을 선택하여 객체를 탐지.
2. **객체 탐지:**탐지된 객체의 클래스 이름, 좌표(xmin, ymin, xmax, ymax), 신뢰도(confidence)를 데이터프레임 형태로 저장.YOLO 결과는 `pandas` 데이터프레임으로 처리하여 간결하게 통계화.
3. **객체 카운팅:**탐지된 각 객체(예: 사람, 책상, 가방 등)의 개수를 통계로 저장.

---
### Google Drive 마운트
      from google.colab import drive
      drive.mount('/content/drive')
      
      # YOLOv5 설치
      !git clone https://github.com/ultralytics/yolov5  # YOLOv5 git repository
      %cd yolov5
      !pip install -r requirements.txt  # 필요한 라이브러리 설치


+ GitHub에서 코드를 클론하고 Google Drive에 데이터를 연결하여 대용량 비디오 데이터를 쉽게 관리 하도록 하였음

  ### 비디오 경로 및 출력 경로 설정
      videopath = '/content/drive/MyDrive/Colab Notebooks/dataset/11월26일 촬영본.mov'
      output_txt_path = '/content/drive/MyDrive/Colab Notebooks/dataset/detection_statistics.txt'
      
      # 비디오 로드
      cap = cv2.VideoCapture(videopath)
      fps = cap.get(cv2.CAP_PROP_FPS)
      frame_skip = 30  # 30프레임마다 (1초 간격으로) 분석
      frame_number = 0  # 프레임 번호 초기화
      
  ### 통계 저장을 위한 데이터 구조
      statistics = defaultdict(list)
      time_intervals = []
      confidence_threshold = 0.7  # 컨센서스 설정


+ 프레임 건너뛰기 설정: 처리 속도를 높이기 위해 30프레임마다 탐지.
  
+ 컨센서스 설정: 잘못된 인식을 낮추기 위해 값 조정


### frame_skip 간격으로만 YOLOv5 모델을 사용하여 객체 탐지
    if frame_number % frame_skip == 0:
        # 현재 타임라인(분:초 형식으로 표시)
        elapsed_time = frame_number / fps  # 경과 시간 (초)
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"  # "MM:SS" 형식으로 변환
        time_intervals.append(time_str)

        # YOLO 모델로 객체 탐지 수행
        results = model(frame)
        detected_objects = results.pandas().xyxy[0]  # Pandas 데이터프레임으로 결과 가져오기

        # 프레임의 감지 통계 수집
        frame_stats = defaultdict(int)
        for _, row in detected_objects.iterrows():
            label = row['name']
            confidence = row['confidence']

            # 신뢰도 기준을 충족한 객체만 기록
            if confidence >= confidence_threshold:
                frame_stats[label] += 1

        # 타임라인별 통계 기록
        for obj, count in frame_stats.items():
            statistics[obj].append(count)
        # 통계가 없는 객체는 0으로 채움
        for obj in statistics:
            if obj not in frame_stats:
                statistics[obj].append(0)

    frame_number += 1
+ YOLOv5를 통해 객체를 감지 및 통계 수집


## 코드 전체 구조
1. **입력 비디오 처리:**
    - 지정된 비디오 파일에서 프레임을 읽어와 YOLO 모델로 분석.
    - `fps`를 기준으로 프레임 간 간격을 설정하여 연산 최적화.
2. **YOLO 모델의 객체 탐지 결과:**
    - 각 프레임에서 탐지된 객체의 클래스와 개수를 기록.
    - 결과를 `defaultdict` 형태로 저장하여 타임라인별 객체 카운트를 관리.
3. **타임라인별 객체 통계 저장:**
    - 각 프레임의 분석 결과를 시간대별(분:초 형식)로 저장.
    - 누락된 객체에 대해서는 `0`으로 채워 통계의 완전성을 보장.
4. **결과 출력:**
    - 객체 탐지 통계 데이터를 시계열 데이터로 변환 가능하도록 준비.
    - `pandas`를 활용하여 데이터프레임 형태로 저장 가능.
    - 추후 시계열 분석 및 시각화를 위해 `.txt` 또는 `.csv` 파일로 출력.

## 4. 결과 및 데이터
- - -
<p align="center">
  <img src="https://github.com/29-min/yolov5--PNP/blob/main/detection_data/11%3A28/detection_statistics_graph(1).png" width="500" height="400"/>
  <img src="https://github.com/29-min/yolov5--PNP/blob/main/detection_data/11%3A28/people.png" width="500" height="400"/>
  <figcaption align="center">11월 28일 결과 데이터</figcaption>
</p>
+ 해당 데이터를 바탕으로 교내 시설에서 요일별, 시간별 사람들의 이동량을 비교할 수 있었음.



## 4. 팀원들의 주제 선정 사유 및 목표
- - -

## 홍다오

### **1) YOLO 모델 선택 이유:**

- YOLO는 단일 네트워크로 이미지 전체를 분석하므로, 속도와 정확도 측면에서 균형이 뛰어남.
- Pre-trained 모델을 사용해 초기 학습 시간 없이 빠른 구현 가능. (COCO dataset 활용)
    
    https://github.com/ultralytics/yolov5
    
    https://blog.naver.com/PostView.naver?blogId=intelliz&logNo=222824372526
    
    https://cocodataset.org/
    
    https://dacon.io/en/forum/405930
    

---

### **산출된 데이터 활용 예시:**

1. 시간대별로 특정 객체(예: 사람)의 혼잡도 변화 관찰.
2. 누적된 데이터로 공간별 피크 타임(혼잡 시간대) 분석.
3. 객체 탐지 데이터와 타임라인을 결합하여 공간 효율성 개선.

### **발전 가능성:**

- 현재 YOLO를 통해 객체 탐지를 완료했으므로, 해당 데이터를 기반으로 시계열 예측(LSTM/GRU)가능.
- 추가적으로 결과를 실시간 시각화하는 대시보드 개발로 이어질 수 있음.

---

### **현재 구현의 한계와 개선 방향:**

### **1) 한계:**

- 객체 탐지의 정확도는 카메라의 품질과 위치에 따라 달라질 수 있음.
- 복잡한 환경(예: 밀집된 공간, 다양한 객체 크기)에서 탐지 정확도가 저하될 수 있음.

### **2) 개선 방향:**

1. **탐지 모델의 미세 조정:**
    - 사전 학습된 YOLO 모델을 Fine-tuning하여 프로젝트 환경에 맞춤화
    - 성별, 의상, 잡화(물건) 등 추가 데이터를 학습하여 부가적인 통계 측정

- - -

## 소은
객체 탐지를 실생활에 유용하게 쓸 수 있는 방법을 고민하다가 평소 저녁시간때 식당에 줄이 길어 오랫동안 기다린 경험을 바탕으로 저녁시간 중 어느 시간때가 제일 유동인구가 많은지를 알고 싶어져 시간별 유동인구수를 측정하는 주제로 선정하게 되었다. 

유동인구수를 측정하는 것에 중점을 두고자 정확하게 사람을 인식할 수 있도록 신뢰도를 70%로 하여 정확히 측정하였고 30분동안 촘촘하게 사람수를 측정하여 어느 시간때에 사람이 몰리는지를 알 수 있었고 이를 시각화하기 위해 그래프와 표를 활용하여 한눈에 알아볼 수 있도록 하였다.

- - -
## 찬미
기숙사 식당을 이용하거나 광장에서 배달 음식을 기다리는 학생들이 겪을 수 있는 차량 충돌 사고를 예방하기 위해, 유동 인구 수를 분석하는 주제를 선정하였다. 

객체 인식 모델로는 빠른 속도와 높은 정확도를 동시에 제공할 수 있는 YOLOv5를 활용했고, 불필요한 object가 탐지되더라도 최대한 사람을 파악할 수 있도록 데이터의 정확도를 높이고자 했다. 통계 자료를 통해 차량과 사람이 가장 많이 탐지되는 시간대를 파악하고, 사고 예방을 위한 기초 통계 자료를 제공하는 것을 목표로 삼았다.

- - -
## 태호
cv를 공부함에 있어 프로젝트를 진행하는 것이 큰 도움이 될 것이라고 생각했다.  cv의 객체탐지를 구현하면서 실생활에 적용할 수 있는 주제에 대해서 생각하였고 사람을 탐지하여 특정 공간의 유동 인구를 파악하는 프로젝트를 선택하게 되었다.

영상 자료를 통해 사람을 탐지하기 위해 정확성이 높다고 알려진 YOLOv5를 사용하였다. 모델이 영상에서 물체를 탐지할 때 사람을 정확하게 사람으로  라벨링하는 것이 중요하다고 생각하였다. 일정 기간 같은 시간에 같은 공간을 촬영한 영상을 통해 그 시간대의 유동 인구에 대한 정보를 얻고 이를 통계로 정리하여 시각화 하였다.

- - -
## 규민
cv가 다방면에 활용이 되고 있어 언젠가 깊게 공부해보고 싶었는데, 마침 좋은 조원들을 만나 교내 복잡도를 분석하는 프로젝트를 진행하게 되었다. 센서나 모터 컨트로를 위주로 하던 로봇과 다르게 학습 모델은 상당히 복잡한 부분이 있었다. 

사용 언어도 달라서 어려움이 있었지만, 다오님의 주도로 많은 공부를 할 수 있어서 좋았습니다. 다들 고생하셨어요!
