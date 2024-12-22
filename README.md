# yolov5--PNP
yolov5를 활용한 교내 이동량 분석
> https://github.com/ultralytics/yolov5

## 1. 기본 개념
- - -
#### a. 영상 처리: 
  디지털 이미지나 비디오에서 유용한 정보를 추출하고 분석하는 것으로,
  CV의 한 분야이다. 이미지, 필터링, 객체 인식, 추적, 분할 등 다양한 작업을 포함하며 이 프로젝트에서는 
  동일한 시간대에 촬영한 영상을 통해 이동량을 분석하는 것을 목표로 하였다.

  
![Object Dection, Localization, Classification](https://github.com/29-min/yolov5--PNP/blob/main/Detection%2C%20Localization%2C%20Classification.png)
#### b. Object Dection: 
  이미지나 비디오 내에서 특정 객체를 식별하고 대항 객체의 위치를 지정하는 기술로
  단순한 분류(Classification)와 달리, 객체의 존재 여부와 함께 위치 정보(bounding box)를 제공하여 정밀 분석이 가능하다.

#### c. Localization: 
  이미지나 비디오 내에서 객체의 위치를 찾는 과정으로, 객체의 경계를 나타내는 bounding box를 생성하고 객체의 크기와 위치를 파악하여 정보를 제공한다.

#### d. Classification: 
  이미지나 비디오 내의 객체를 특정 카테고리로 분류하는 작업. 이미지 내의 객체가 사람, 자동차, 고양이 등 어떤 클래스에 속하는지 식별한다. 

#### e. YOLO(You Only Look Once): 
  객체 인식을 실시간으로 수행할 수 있는 딥러닝 모델로 이미지를 X * X 그리드로 분할, 각 그리드 셀이 bounding box와 클래스의 확률을 예측함. 이 과정은 단일 신경망을 통해 한 번에 이루어지므로 속도가 빠르다는 이점이 있다. 

#### f. R-NN 대비 YOLO가 갖는 이점:
  R-CNN(Regin-based Convolutional Nerual Networks)은 객체 검출에서 높은 정확도를 제공하지만, 여러 단계로 이뤄진 층으로 구조가 복잡하고 속도가 느리다. YOLO는 단일 신경망이며, 전체 이미지를 고려하여 예측하기 때문에 전반적인 컨텍스트를 반영한 정확한 검출이 가능하다.

## 2. 주요 코드 설명

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
  
## 3. 결과 및 데이터
- - -
![결과 그래프](https://github.com/29-min/yolov5--PNP/blob/main/detection_data/11%3A28/detection_statistics_graph(1).png)
![결과 꺾은선 그래프](https://github.com/29-min/yolov5--PNP/blob/main/detection_data/11%3A28/people.png)

