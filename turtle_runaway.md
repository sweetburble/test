# Turtle Runaway

사용자는 파란색 도망치는 runner 거북이가 됩니다.  
빨간색 chaser 거북이는 랜덤하게 움직이며, 제한시간동안 잡히지 않는 것이 목표입니다.  

### 1. setter는 "iscatched, 남은시간, 현재 점수"의 "글자"만 보여주는 turtle 객체입니다.

### 2. drawer는 iscatched의 값만 0.1초마다 갱신해서 출력하는 turtle 객체입니다.

### 3. timer는 남은 시간의 값만 0.1초마다 갱신해서 출력하는 turtle 객체입니다.

### 4. score는 현재 점수의 값만 0.1초마다 갱신해서 출력하는 turtle 객체입니다.

# Mandatory
## 1. Add a timer
시작할 때, setter turtle이 초기화되며 "Is catched?, 남은 시간, 현재 점수"을 화면에 계속해서 띄웁니다.
![](%EC%B4%88%EA%B8%B0%ED%99%94%EB%A9%B4.PNG)
남은 시간은 total_time(기본 30초)에서 now만큼 빼서 cur_time에 저장합니다.

## 3. Add your concept of score
total_score는 시간이 지날수록 증가하며 총 100점입니다.

## 2. Add your intelligent Turtle
chaser Turtle은 각각 20%의 확률로 좌회전,우회전을 하고, 60%는 직진을 하도록 했습니다.

거기에 xcor(),ycor()로 chaser turtle의 좌표를 주기적으로 감시하여 chaser가 맵 밖으로 나가려고 할때, 180도 회전한 다음 한 번 전진하게 하였습니다.


### 마지막으로 만약 is_catched가 True이거나 남은 시간이 0보다 작다면, 현재 점수를 5초간 출력하고 게임을 종료합니다.
![](%EC%9E%A1%ED%9E%8C%20%EC%83%81%ED%99%A9.PNG)