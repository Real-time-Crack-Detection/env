# 진행사항 - 최현웅
### Part : 실시간 탐지 모델 구현
* * *
## 2020.02.25
- keras 기반 faster rcnn code 확인 후, 깃헙에 commit 정상적으로 완료.
- 현재 해당 코드는 imagenet, 이전 버젼 keras에 맞게 커스터마이징 되어 있어, 해당 코드를 좀 더 살펴보며 수정해 나갈 예정.

## 2020.02.28
- R-CNN, Fast R-CNN, Faster R-CNN간의 연관성 및 차이점 분석.
- Faster R-CNN에서 사용되는 Pooling 방식인 Spatial Pyramid Pooling에 대한 논문 분석.

## 2020.03.01
- R-CNN, Fast R-CNN, Faster R-CNN의 특징 정리
- Faster R-CNN를 Train 시키기 위한, 자료 및 프로그램 정리

## 2020.03.06
- 실제 크롤링한 데이터로 Train set 파일 생성
- 총 클래스는 4개로 하도록 함 -> 0: crack, 1: horizon crack, 2: vertical crack, 3: step crack
- 0 = 일반 crack으로 주로, 대각선의 형태 및 트리의 형태를 띔.
- 1 = horizon crack은 해당 부분이 노후 되어 윗 부분의 무게를 이기지 못하고 발생한 균열로, 크다면 위험하지만 일반적으로 2보다는 덜 위험함.
- 2 = vertical crack은 심각한 균열로, 건물 붕괴와 같은 재난으로 이어질 수 있음.
- 3 = 주로 벽돌집에서 나타나는 계단형 균열

