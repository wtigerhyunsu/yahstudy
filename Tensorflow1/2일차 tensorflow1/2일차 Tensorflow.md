## 변수 (variable)

달라 질수 있는 값 

![image-20200814221510396](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200814221510396.png)

표에서 column 을 변수라고 한다 .

![image-20200814221708097](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200814221708097.png)

> 온도와 판매량 이  관련이 있어 보인다. 하나의 표안에는 두가지가 모두 표현된다. 지도학습은 이 두가지를 구분하는 것에서 시작된다. 

## Pandas 

사용법

> imoprt pandas as pd 
>
> 데이터를 불러오고 데이터를 분리하는 것을 할수있다 실습에 이용한다 

![image-20200814222016986](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200814222016986.png)

https://github.com/blackdew/tensorflow1/tree/master/csv >>>> 코드 

## 실습을 통해 배울 도구 

1. 파일 읽어 오기 
2. 모양 확인하기
3. 칼럼 선택하기
4. 칼럼이름 출력하기
5. 맨위 5개 관측치 출력하기

> [[https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv ](https://www.youtube.com/redirect?redir_token=QUFFLUhqbjJTUFFTd0hRWjR0Umx2UkRFVGNvdE5Ca05WZ3xBQ3Jtc0trRHppT01LVUVmTEtJbTZzV0VCSEhBbFpCMVBVUnFoc3FxRWRWWmdtbDljVXZlLWpycU5fNFg1VDBZR29XR2labUxiMjdlaklFMnZ4S2tMbWFWd0F2VWQyY0RPcDdZekh3UUpXN0JJdHA2RU5iX1BxNA%3D%3D&q=https%3A%2F%2Fraw.githubusercontent.com%2Fblackdew%2Ftensorflow1%2Fmaster%2Fcsv%2Flemonade.csv&stzid=Ugygk-_byVfYAFKSPVx4AaABAg&event=comments) [https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv ](https://www.youtube.com/redirect?redir_token=QUFFLUhqbmFsbXdpTkVuV1pQaU1FcDYzakxEbHZVaW1LZ3xBQ3Jtc0tubWJwcUotM2N5VGRld1QzZnFfTDB6QUhjYnZlOVZac0dBRU52eVotVzM4ZlZKLXJxUktReFlrVU43QjRuQ1VfQ1RPbTFkV0VoQ0VXaTJMZnQwOWNlUWJtX2Y0UDNyX1VHTFVvRnVZUnliQ1RZaFloWQ%3D%3D&q=https%3A%2F%2Fraw.githubusercontent.com%2Fblackdew%2Ftensorflow1%2Fmaster%2Fcsv%2Fboston.csv&stzid=Ugygk-_byVfYAFKSPVx4AaABAg&event=comments) [https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv]

![image-20200814223033500](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200814223033500.png)

# 데이터 모양으로 확인하기 
```python
print(레모네이드.shape)
print(보스톤.shape)
print(아이리스.shape)

#칼럼이름 출력
print(레모네이드.columns)
print(보스톤.columns)
print(아이리스.columns)

독립 = 레모네이드[['온도']]
종속 = 레모네이드[['판매량']]
print(독립.shape, 종속.shape)

독립 = 보스톤[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']]
종속 = 보스톤[['medv']]
print(독립.shape, 종속.shape)

독립 = 아이리스[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
종속 = 아이리스[['품종']]
print(독립.shape, 종속.shape)

레모네이드.head() ##맨위에서 5개 만 가져온다 

보스톤.head()

아이리스.head()
```





### 지도 학습 방법

![image-20200814224448972](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200814224448972.png)

![image-20200814224903654](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200814224903654.png)

> 매칭 시켜본다 
>
> 우리가 준비한 독립 변수  종속 변수의 개수가 들어간다 input 에 독립변수가 Dense 에 종속 변수가 들어간다 

![image-20200814225108072](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200814225108072.png)

> epochs 는 몇번 반복해서 학습할지를 결정 해주는 숫자 
>
> 학습이 끝나고 15 라는 변수를 넣었을 떄 값을 얻을수 있다 

## model.fit(독립,종속 epochs=10) 

![image-20200814225436966](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200814225436966.png)

> 10 번 반복해서 학습하라고 제시한 코드 
>
> 파란색은 몇번째 학습인지 알려주는 부분
>
> 주황색은 학습마다 얼마나 시간이 걸렸는지 알려주는 부분
>
> 초록색은 학습이 얼마나 진행되었는지 알려주는 부분 
>
> 모델은 매학습마다 조금식 학습을 하게 되는데 얼마나 정답에 가까이 맞추고 있는 지 평가하는 지표 입니다.
>
> loss 는 (예측 - 결과)의 제곱 을 평균을 냈을때 0이 되면 정답에 가까운것이다

# 레모네이드 판매 예측

loss 확인 결과

![image-20200814231341498](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200814231341498.png)

> verbose =0 을 위에 붙여 주면 학습 내용이 출력 x 

```python
#라이브러리 사용
import tensorflow as tf
import pandas as pd

#데이터 준비
fileurl = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
data = pd.read_csv(fileurl)
data.head()

#종속병수,독립변수
독립 = data[['온도']]
종속 = data[['판매량']]
print( 독립.shape, 종속.shape)

#모델을 만듭니다.
x = tf.keras.layers.Input(shape=[1])
y = tf.keras.layers.Dense(1)(x)
model = tf.keras.models.Model(x,y)
model.compile(loss='mse')

#모델을 학습합니다.
model.fit(독립, 종속, epochs=1000, verbose =0)
model.fit(독립, 종속, epochs=10)

#모델을 이용합니다 
model.predict(독립)

model.predict([[15]])
```

