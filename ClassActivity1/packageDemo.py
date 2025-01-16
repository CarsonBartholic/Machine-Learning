import matplotlib
matplotlib.use('TkAgg')  # Gets rid of errors when stopping the program

import numpy as np
from sklearn.datasets import load_iris 
import matplotlib.pyplot as plt

iris = load_iris()

x_axis = 3
y_axis = 2

_, ax = plt.subplots()
scatter = ax.scatter(iris.data[:, x_axis], iris.data[:, y_axis], c = iris.target) #pulls all data
ax.set(xlabel=iris.feature_names[x_axis], ylabel=iris.feature_names[y_axis]) #shows axis labels
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes" #creates legend
)


average_of_y = np.mean(iris.data[:, y_axis]) #average length
average_of_x = np.mean(iris.data[:, x_axis]) #average width
ratio = average_of_y / average_of_x #ratio
print("Average of" , iris.feature_names[y_axis], "= ", average_of_y)
print("Average of" , iris.feature_names[x_axis], "= ", average_of_x)
print("Average of" , iris.feature_names[y_axis] , "/ Average of" , iris.feature_names[x_axis] , " = " , ratio)


plt.show()