# 프로젝트 결과 보고서

#### 주제 : 최근 생성형 AI로 인해 범람하는 피싱사이트의 탐지
#### 팀원 : 김영서, 정태현


## 1. 프로젝트 문제 정의
2022년 말부터 GPT-3.5를 위시한 ChatGPT가 대중에게 알려지고, 유명해졌다.
ChatGPT는 질문에 답을 해주거나, 정보 검색, 조작에 뛰어난 성능을 자랑했으나, 이런 생성형 인공지능(AI)이 만든 불필요한 정보나 정확하지 않은 정보로 만들어진 콘텐츠들이 넘쳐나면서 인터넷을 오염시키고 있다는 지적이 나왔다.

실제로 뉴스사이트 평가회사인 뉴스가드는 올해 5월 초 AI를 활용해 콘텐츠를 제작하는 가짜뉴스 웹사이트 49개를 발견했다고 밝혔다. 이후 6월 말 기준으로는 가짜뉴스 웹사이트가 277개로 급증했다.
동영상 사이트 유튜브에서도 챗GPT 관련 콘텐츠가 급증했다. 문제는 ‘챗GPT로 일주일 만에 수천 달러를 벌 수 있다’ 같은 자극적인 내용으로 대중들을 현혹시키는, 이른바 ‘정크 콘텐츠’가 넘쳐나는 것이라고 생각한다.
또한 근본적으로 피싱 사이트는 신뢰할 수 있는 사람 또는 기업인 것처럼 가장함으로써, 
비밀번호, 신용카드 정보, 금융 정보와 같이 기밀을 요하는 정보를 얻으려는 사회공학적 해킹 기법이다.

따라서 이 프로젝트는 정상 사이트와 피싱 사이트의 많은 데이터를 수집하고, 거기에 머신러닝을 사용함으로써 최근 생성형 AI로 인해 범람하는 피싱사이트를 탐지하고, 또 개인 정보 유출 및 금융피해 발생을 최소화하고자 한다.

<img src="https://drive.google.com/uc?export=view&id=15gJzpd_7SUK7ef9-IiBxe1zzUCjNV4VR" width="400" height="400">
<img src="https://drive.google.com/uc?export=view&id=13W9qK5ZKaAH3tH4RYrIAQudZ38rvk01k" width="400" height="400">

## 2. 데이터에 대한 이해 및 전처리
정상 사이트 Dataset은 https://majestic.com/reports/majestic-million 에서 
인기순으로 정렬된 정확히 100만개의 최신 정상 사이트 데이터를 얻을 수 있었다. 
이 사이트에서 제공된 데이터에는 단순한 도메인 뿐만이 아니라 “GlobalRank”, “RefSubNets”, “RefIPs”, “PrevGlobalRank”, “PrevRefSubNets” 등 머신러닝에 필요하지 않은 속성이 많았다.
그리고 크롤링한 피싱 사이트 데이터와는 다르게 정상 사이트라는 label이 붙어있지 않았다.
따라서 pandas를 사용해서, 필요 없는 속성은 제거하고 label이 들어갈 ‘Phish’ 속성에 ‘INVALID’ 라는 값을 추가했다.
 
피싱 사이트 Dataset은 https://phishtank.org/ 에서 
파이썬의 selenium을 이용한 웹 크롤링으로 4만개의 최신 피싱 사이트 데이터를 얻을 수 있었다.

```python
urls_data = pd.read_csv('./datasets/Phishing_URLS_dataset/all_data.csv')
urls_data.isnull().sum() # null data가 있는지 확인했다.

count = urls_data['Phish'].value_counts()
count # 성공적으로 ‘INVAILID’한 정상 데이터 100만개와 ‘VALID PHISH’한 피싱 데이터 4만개가 저장되었다.
```

### 2-1. 데이터 전처리
우리가 가지고 있는 데이터의 유효한 정보는 url 하나로, 문자열은 categorical variable이기 때문에 데이터 전처리가 필수였다.

#### 1) Tokenizing
Tokenizing이란, 문자열을 여러개의 조각(Token)으로 쪼개는 것이다.
여기서 Token이란 문자열의 한 조각으로, '단어 Token'이나 '문장 Token' 등으로 분리한다.
특히 URL에서 '://', '@', '?' 등의 특수문자를 제거하기 위하여 토큰화(tokenize) 과정을 거쳤다.
NLTK에서 여러 가지 Tokenizer가 있지만, 정규표현식을 이용하여 토큰화하는 RegexpTokenizer를 사용했다.
```python
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
tokenized_url_example = tokenizer.tokenize(urls_data['Domain'][0])
print(tokenized_url_example) # ['https', 'google', 'com']
```

#### 2) Stemming
Stemming은 어형이 변형된 단어로부터 접사 등을 제거하고 그 단어의 어간을 분리해 내는 것을 의미한다.
Stemming은 검색엔진에서 색인할 때 가장 많이 쓴다. 모든 형태의 단어를 저장하는것 보다 Stemming을 거친 단어를 저장하는 것이 색인 크기를 줄일 뿐만 아니라, 검색 정확성을 높일 수 있다.

nltk에서 제공하는 PorterStemmer, LancasterStemmer, SnowballStemmer 중 
PorterStemmer는 너무 보수적인 방식이라 완벽하게 추출하기는 힘들고, 
SnowballStemmer를 쓰기에는 url은 이미 영어나 숫자로 통일되어 있으므로 필요성이 낮다고 생각했다.
따라서 LancasterStemmer를 Stemming에 사용했다.
```python
stemmer = LancasterStemmer()
stemmed_url_example = [stemmer.stem(word) for word in tokenized_url_example]
print(stemmed_url_example) # ['https', 'googl', 'com']

# 모든 stemmed한 word를 join한다
sent_url_example = ' '.join(stemmed_url_example)
print(sent_url_example) # https googl com
```

#### 3) Vectorizing
머신러닝 분야에서, Vectorizer는 주로 텍스트를 쉽게 분석하기 위해 벡터로 표현할 때 사용한다.
숫자나 벡터를 입력값으로 기대하는 여러 머신러닝 모델을 실행하기 위해서는 텍스트나 또 다른 형태의 데이터를 숫자나 벡터로 나타낼 필요가 있고, 이 때 사용하는 것이 바로 Vectorizer이다.

역시 Vectorizer에는 주로 CountVectorizer나 TfidfVectorizer 를 사용하는데, CountVectorizer는 
문서 집합(문서 리스트)에서 단어 토큰을 생성하고 각 단어의 수를 세어 BOW(Bag of Word) 기반으로 벡터를 만든다.

TfidVectorizer는 TF-IDF 방식으로 단어의 가중치를 조정한 BOW 벡터를 만든다. 
TF(Term Frequency) : 특정 단어의 빈도수
DF(Document Frequency) : 특정 단어가 들어가있는 문서의 수 => IDF : DF의 역수이다.
즉, TF-IDF를 이용하면 많은 문서에 등장하는 단어는 비중이 작아지고, 특정 문서군에서만 등장하는 단어는 비중이 높아진다고 할 수 있다. 학습 결과, 약간 더 precision이 높은 TfidfVectorizer를 사용하였다.

```python
cv = TfidfVectorizer()
preprocessed_url_data_example = cv.fit_transform([sent_url_example])
preprocessed_url_data_example.toarray() # array([[0.57735027, 0.57735027, 0.57735027]])
```

그리고 피싱 사이트 데이터에 비해(4만개), 정상 사이트 데이터가 너무 많아서(100만개) 학습 도중 데이터의 불균형이 발생할 수 있다고 판단 되었다. 
따라서, 정상 사이트 데이터를 랜덤하게 추출하는 preprocessing 메서드를 따로 정의했다. 

```python
def Domain_URL_Preprocessing(urls_data, num=None): # num 값으로 INVALID data 개수를 조절한다
    if num is not None:
        if num >= 0 and num < 1000000:
            urls_data_invalid = urls_data.loc[urls_data['Phish'] == 'INVALID']
            urls_data_valid = urls_data.loc[urls_data['Phish'] == 'VALID PHISH']
        
            urls_data_invalid_sample = urls_data_invalid.sample(n=num, random_state=34)
            urls_data = pd.concat([urls_data_invalid_sample, urls_data_valid], ignore_index=True)
    
    urls_data['Domain_Tokenized'] = urls_data['Domain'].map(lambda d : tokenizer.tokenize(d))
    urls_data['Domain_Stemmed'] = urls_data['Domain_Tokenized'].map(lambda l : [stemmer.stem(word) for word in l])
    urls_data['Domain_sent'] = urls_data['Domain_Stemmed'].map(lambda s : ' '.join(s))
    preprocessed_urls_data_domain = cv.fit_transform(urls_data['Domain_sent'])
    return urls_data, preprocessed_urls_data_domain

urls_data, preprocessed_urls_data_domain = Domain_URL_Preprocessing(urls_data, 40000)
```

#### 4) 정답 label 인코딩
'VALID Phish'와 'INVALID' 라는 두 가지 범주로 구성된 label을 인코딩하기 위해서 LabelEncoder를 사용했다.
OneHotEncoder는 각 범주를 독립적인 이진 변수로 변환한다. 범주 간에 순서나 중요도가 없는 경우에 적합하다.
OrdinalEncoder는 범주를 정수로 변환하지만, 여러 범주가 있는 경우에 순서가 유지된다. 
즉, 범주 간에 순서나 중요도가 있는 경우에 적합하다. 

LabelEncoder는 binary classification 문제에서 레이블을 인코딩하기에 적합한 방법이다. 
여기서는 'VALID Phish'를 1로, 'INVALID'를 0으로 변환하였다.
```python
encoder = LabelEncoder()
urls_data_encoded_labels = encoder.fit_transform(urls_data_labels)
urls_data_encoded_labels[:10] # array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

class_names = ['INVALID', 'VALID PHISH']
print(class_names[urls_data_encoded_labels[0]]) # INVALID
print(class_names[urls_data_encoded_labels[-1]]) # VALID PHISH
```

#### 5) training set, test set 분리
test set은 절대로 학습에 활용되지 않고, 성능 측정도 마지막에 딱 한번만 함으로써 독립성을 보장해야 한다.

```python
X_train, X_test, y_train, y_test = train_test_split(preprocessed_urls_data_domain, urls_data_encoded_labels, 
                                                   test_size=0.2, random_state=34, shuffle=True, 
                                                    stratify=urls_data_encoded_labels)
```
test_size=0.2 : 전체 데이터의 20%가 테스트 데이터로 사용된다.
random_state=34 : 데이터 분리 시 사용되는 난수 생성기의 시드값이다. 
이 값을 설정하면 항상 같은 방식으로 데이터가 분리되어, 실험의 재현성을 보장한다.
shuffle=True : 데이터를 분리하기 전에 데이터를 무작위로 섞을지를 결정한다.
stratify=urls_data_encoded_labels : label 데이터의 클래스 비율을 
학습용 dataset과 테스트용dataset에서 동일하게 유지하기 위해 계층적으로 데이터를 분리한다. 
여기서는, 원본 dataset에서 'VALID PHISH'가 차지하는 비율이 50%, 'INVALID'가 차지하는 비율이 50%니까, 학습용 dataset과 테스트용 dataset에서도 동일한 비율을 유지한다.



## 3. 모델 선택 및 이론적 배경
이 프로젝트에서는 Stochastic Gradient Descent 알고리즘을 사용한 sklearn.linear_model 라이브러리의 SGDClassifier 모델을 선택했다.

이전에 테스트했던 다른 알고리즘과 비교하였을 때, 선택하게 된 이유는 다음과 같다.
#### 1) Support Vector Machine
sklearn.svm의 SVC 클래스는 커널 트릭을 지원하지만, training하는 시간 복잡도는 O(m2 x n)에서 O(m3 x n) 사이이다. 
그런데 이 프로젝트는 최소 8만개에서 ~ 하기에 따라 최대 104만개의 데이터 인스턴스를 갖고 있다.
즉, 학습하는 데 시간이 너무 오래 걸렸고, SVM 알고리즘은 "복잡하지만, 중소규모"인 data set에 적합하다.

#### 2) RandomForest
RandomForest는 여러 개의 독립적인 트리(decision tree)를 학습하여 그 결과를 집계하는 방식으로 동작한다. 
이는 Bagging(Bootstrap Aggregating)이라는 앙상블 기법을 사용하며, 각 트리는 원본 데이터에서 무작위로 선택된 subset을 이용해 학습된다. 
이로 인해 모델은 Overfitting을 피하고, 일반화 성능을 향상시키는 효과를 얻을 수 있다.

▶ 처음, 정상 사이트 데이터 4만개 : 피싱 사이트 데이터 4만개로 학습했을 때는 
Accuracy: 0.8808, Precision: 0.9200, Recall: 0.8342, F1: 0.8750정도로 우수한 성능을 보였다.

▶ 하지만 정상 사이트 데이터가 10만개로 증가하자
Accuracy: 0.7159, Precision: 1.000, Recall: 0.0057, F1: 0.0114 로 성능이 크게 감소했다.
즉, 이때는 10만개의 정상 사이트는 100% 완벽하게 정상 사이트로 예측을 했으나, 
4만개의 피싱 사이트 중 0.57% 정도만 실제로 피싱 사이트로 예측하고, 나머지 99% 가량 모두 정상 사이트로 판단했다는 것이다!

▶ 모든 데이터를 사용한 상황에서는 
Accuracy: 0.9615, Precision: 0.000, Recall: 0.000, F1: 0.000 이라는 처참한 성능을 보여주었다.
즉, 104만개의 모든 데이터를 정상 사이트라고 예측했으며, 아예 classifier의 제 역할을 하지 못했다.

이 결과는 아마도 **불균형한 dataset에서 비롯된다**고 생각했다.
dataset에서 positive 클래스와 negative 클래스의 비율이 크게 차이날 경우, 
모델은 다수 클래스에 치우쳐 학습하게 될 수 있다. 이 경우, 다수 클래스(여기서는 negative)에 대한 예측은 잘하나 소수 클래스(여기서는 positive)에 대한 예측은 잘 못하는 현상이 발생하게 된 것이다.

#### 3) Logistic Regression
로지스틱 회귀 같은 일부 회귀 알고리즘을 classification에도 사용할 수 있다. 
로지스틱 회귀 모델은 선형 회귀 모델처럼 결과를 직접 출력하는 대신,
입력과 가중치의 선형 조합을 로지스틱 함수를 통해 확률로 변환하고, 이를 기반으로 클래스를 예측한다.
Maximum Likelihood Estimation을 사용하여 모델을 학습하며, decision boundary는 선형이다.

테스트 결과, 전체적인 성능은 SGDClassifier와 비슷하게 나오지만, SGD 알고리즘이 대용량 데이터를 처리하는데 유리하고, 다양한 regularization 옵션이나, 다양한 loss function 옵션을 제공하므로 선택하지 않았다.

### ■ Stochastic Gradient Descent 알고리즘과 SGDClassifier
SGD는 경사 하강법의 변형으로, 전체 dataset 대신 랜덤하게 선택한 하나 또는 일부의 데이터를 사용하여 각 step에서 기울기(Gradient)를 계산한다.

SGD를 선택하게 된 장점은 다음과 같다.
##### 장점 1. 한 번에 하나의 인스턴스으로만 학습하면, 조작할 데이터가 거의 없기 때문에 알고리즘이 훨씬 빨라진다.
##### 장점 2. 또한 각 반복마다 하나의 인스턴스만 메모리에 저장하면 되므로, 대규모 학습 세트에 대한 학습도 가능하다.
##### 장점 3. 또한 무작위적 특성 때문에 지역(local) 최소값에서 벗어나, 전역(global) 최소값을 찾는 데 도움이 된다.

**무작위적 특성으로 인해 batch 경사 하강법보다 훨씬 덜 규칙적이라는 단점**도 있기는 하다.
즉, 최소값을 향해 완만하게 감소하는 대신, cost function이 위아래로 뛰어 오르내리면서 평균적으로만 감소한다.



## 4. 모델 학습 및 튜닝
```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(penalty='l2') # overfitting을 막기 위한 penalty는 L2 Regularization (Ridge)를 사용
sgd_clf.fit(X_train, y_train)

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=10) 
```
Cross Validation 기법을 사용했다. 학습 dataset을 10개의 폴드로 나누어서, 
9개 폴드는 그대로 모델을 학습하고, 나머지 1개의 폴드는 validation(학습 중 테스트)에 사용한다. 
이 과정을 총 10번 반복해서, 각각의 데이터 샘플이 정확히 한 번씩 테스트에 사용되도록 한다.
-> 이렇게 교차 검증을 통해 생성된 예측값(y_train_pred)을 사용하면, 모델의 성능을 보다 객관적으로 평가할 수 있다.



## 5. 모델 성능 평가
```python
cf_matrix = confusion_matrix(y_train, y_train_pred)
sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, 
                 fmt='.2%', cmap='Blues')
```
이렇게 confusion matrix를 생성하고 시각화하니까, 이 SGD모델이 어떤 클래스를 잘 예측하고,
어떤 클래스는 잘못 예측하는지 쉽게 파악할 수 있었다. 또한, 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 Score같은 다양한 성능 지표를 계산하는 데 사용할 수도 있다. 


Cross validation으로 얻은 classification 결과는 다음과 같았다.
```python
print(accuracy_score(y_train, y_train_pred)) # 0.9550…
print(precision_score(y_train, y_train_pred)) # 0.9850…
print(recall_score(y_train, y_train_pred)) # 0.9185…
print(f1_score(y_train, y_train_pred)) # 0.9505…
```
accuracy가 가장 흔하고 직관적인 성능 기준이지만, 왜곡된 dataset일 때는 적합하지 않다. 
(하지만, 이 프로젝트에서는 사용하는 데이터가 1:1 비율로 accuracy도 합리적인 성능 기준이다)

게다가, Trade-off 관계인 precision과 recall의 지표 또한 98.5%와 91.8% 정도로 뛰어나고, 
둘 다 골고루 높으니까 F1 Score도 약 0.95로 모든 성능 지표가 우수하다는 것을 볼 수 있었다.


그 다음은 ROC curve를 그리고, 그 아래의 면적(AUC)를 계산하는 과정이다.
```python
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate(recall)')
y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=10, method="decision_function")

fpr, tpr, thresholds = roc_curve(y_train, y_scores)
plot_roc_curve(fpr, tpr)
plt.show()

print(roc_auc_score(y_train, y_scores)) # 0.9845…
```
PR curve 대신 ROC curve를 사용한 이유는 이 프로젝트의 dataset의 클래스들이 불균형하지 않기 때문이다.
따라서 ROC curve와 그 AUC가 온존히 성능을 보여주는 지표가 된다.

ROC curve는 가로축이 FPR(False Positivie Rate)이고, 세로축이 TPR(Recall)인 곡선이다.
순수한 랜덤 분류기는 y = x인 직선이고, 완벽한 분류기일수록 왼쪽 상단에 곡선이 가까워진다.

뿐만 아니라, ROC 곡선 아래의 면적인 (AUC)를 계산하고, 그 결과도 출력했다.
순수한 랜덤 분류기는 ROC AUC가 0.5이고, 완벽한 분류기일수록 ROC AUC가 1에 가까워진다.
AUC가 0.9845로, 1에 정말 가까운 것을 알 수 있었다.


마지막으로는 처음에 분리했던 test data로 모델의 최종 성능을 평가했다.
```python
y_pred = sgd_clf.predict(X_test)
print("Accuracy: %.8f" % accuracy_score(y_test, y_pred)) # Accuracy: 0.9526…
print("Precision: %.8f" % precision_score(y_test, y_pred)) # Precision: 0.9839…
print("Recall: %.8f" % recall_score(y_test, y_pred)) # Recall: 0.9203…
print("F1: %.8f" % f1_score(y_test, y_pred)) # F1: 0.9511…
```

처음 정상 사이트의 데이터를 100만 개 모두 다 사용했을 때는 Accuracy는 0.9864… (98.64%) 로 매우 우수했지만, Precision은 0.9603으로 조금 낮았고, 특히 Recall은 0.6743으로 매우 낮았다.

특히, 이번 프로젝트의 주제는 “피싱 사이트 탐지”이므로, 
실제로 피싱 사이트인 데이터를 모델이 잘 예측할수록(Recall이 높을수록) 만족도가 높다고 생각했다.
따라서 dataset 클래스의 비율이 [100 : 4], [10 : 4], [1 : 1]으로 나뉘어진 3가지의 모델 중에서
Recall을 비롯해서 가장 성능이 우수한 [1 : 1] (40000 : 40000) 모델을 사용했다.



## 6. 코드 설명?



## 7. 결론 및 향후 방향
1) 기존의 방식을 고집하지 않고 머신러닝 기술을 활용함으로써, 더욱 높은 피싱 사이트 탐지율을 기대할 수 있고
공공기관, 금융기관 사칭을 통한 개인정보 유출 피해를 방지하고자 한다.

2) 또한 범죄 기술 발전에 따라 더욱 정교해진 피싱 범죄를 머신러닝, 딥러닝 기술을 통해 사전에 필터링 할 수 있다.

3) 향후에는 이 프로젝트를 “브라우저 확장 프로그램”이나 “탐지 애플리케이션”으로 발전시켜서, 
생성형 AI를 활용해서 무분별하고, 알맹이는 없는 사이트, 도메인, 블로그 글로 위장한 피싱 사이트 등이
넘쳐나는 인터넷에서 일종의 이정표가 되었으면 한다.



## 8. 참고 문헌
[영남이공대학교_CSS(Cyber Security Specialist)_머신러닝을 활용한 피싱사이트 탐지 연구 및 탐지 앱 개발]

[NLTK에서의 Tokenizer의 종류] https://dbrang.tistory.com/m/1151

[Stemming, Lemmatizing] https://excelsior-cjh.tistory.com/67

[Vectorizer in scikit-learn] https://17th-floor.tistory.com/4
