from sklearn.linear_model import LinearRegression
import pandas as pd

train = pd.read_csv("/Users/baikai/Downloads/data/train.csv")
test = pd.read_csv("/Users/baikai/Downloads/data/test.csv")
submit = pd.read_csv("/Users/baikai/Downloads/data/sample_submit.csv")

train.drop('id', axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)


y_train = train.pop('y')

reg = LinearRegression()
reg.fit(train, y_train)
y_pred = reg.predict(test)

y_pred = list(map(lambda x : x if x >= 0 else 0, y_pred))

submit['y'] = y_pred
submit.to_csv("/Users/baikai/Downloads/data/my_LR_prediction.csv", index=False)
