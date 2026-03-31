import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Score": [2,4,5,6,7,8,8.5,9]
}

df = pd.DataFrame(data)

print(df)

X = df[["Hours"]]
y = df["Score"]

model = LinearRegression()
model.fit(X, y)

new_hours = [[6]]
predicted_score = model.predict(new_hours)
print("\nPredicted score:", predicted_score[0])

test_data = [[4], [6], [9]]
predictions = model.predict(test_data)

print("\nMultiple predictions:")
for i in range(len(test_data)):
    hours = test_data[i][0]
    score = predictions[i]
    
    print(f"{hours} giờ -> {score:.2f} điểm")

plt.scatter(X, y)
plt.plot(X, model.predict(X))

plt.xlabel("Hours")
plt.ylabel("Score")
plt.title("Hours vs Score")

plt.show()

y_pred = model.predict(X)
score = r2_score(y, y_pred)
print("\nR2 Score:", score)