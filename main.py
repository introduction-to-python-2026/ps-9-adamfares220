import pandas as pd

df = pd.read_csv("parkinsons.csv")
df = df.dropna()
df.head()

print(df.columns.to_list())

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue="PPE", diag_kind="kde", corner=True)
plt.show()

selected_features = ['status', 'HNR']
x = df[selected_features]
y = df['PPE']

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train, y_train)

from sklearn.metrics import r2_score

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(f'R2 Score: {r2}')

import joblib

joblib.dump(model, 'my_model.joblib')
