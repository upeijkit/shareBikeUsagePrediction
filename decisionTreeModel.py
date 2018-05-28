from sklearn.tree import DecisionTreeRegressor
import pandas as pd

train = pd.read_csv("/Users/baikai/Downloads/data/train.csv")
test = pd.read_csv("/Users/baikai/Downloads/data/test.csv")
submit = pd.read_csv("/Users/baikai/Downloads/data/sample_submit.csv")


train.drop('id', axis=1, inplace=True)
test.drop('id',axis=1,inplace=True)

y_train = train.pop('y')

reg = DecisionTreeRegressor(max_depth=5)
reg.fit(train, y_train)
y_pred = reg.predict(test)

submit['y'] = y_pred
submit.