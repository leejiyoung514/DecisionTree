from sklearn import datasets
import sklearn.model_selection
import sklearn.preprocessing
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydotplus
import graphviz
from IPython.display import Image

iris = datasets.load_iris()
print(iris.keys())

print(iris['DESCR'][:193] + "\n...")
print('타깃의 이름:{}'.format(iris['target_names']))
print("특성의 이름:{}".format(iris['feature_names']))
print("data의 크기:{}".format(iris['data'].shape))
print('data의 처음 다섯 행:\n{}'.format(iris['data'][:5]))

X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print(X_train[:5])
print(X_train_std[:5])

iris_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3,
                                   random_state=0)
iris_tree.fit(X_train, y_train)
print("학습용 데이터넷 정확도:{:.3f}".format(iris_tree.score(X_train, y_train)))
print("검증용 데이터넷 정확도:{:.3f}".format(iris_tree.score(X_test, y_test)))

dot_data = export_graphviz(iris_tree, out_file=None, feature_names=['petal length', 'petal width'],
class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
