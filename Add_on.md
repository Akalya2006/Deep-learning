code:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
X= np.array([[0,0], [0,1], [1,0], [1,1]])
y= np.array([0,1,1,0))
c1f Perceptron(tol-1e-3, random_state=0)

clf.fit(X, y)
print("Predictions:", clf.predict(X))
for i in range(len(X)):
      if y[i]=0:
            plt.scatter(X[i][0], X[i][1], color='red')
      else:
            plt.scatter(X[1][0], X[i][1], color='blue')
x_values =[0, 1]
y_values=- (clf.coef_[0][0]"np.array(x_values) clf.intercept_)/clf.coef_[0][1]
plt.plot(x_values, y_values)
plt.title('Perceptron Decision Boundary for XOR')
plt.show()

output![output 1 dp](https://github.com/user-attachments/assets/bd67c9f7-7fea-46e9-9d7f-24f29b441c01)
