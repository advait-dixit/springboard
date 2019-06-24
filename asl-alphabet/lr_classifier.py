from PIL import Image
import itertools
from sklearn import linear_model, metrics, model_selection

X_train = []
y_train = []
X_test = []
y_test = []
for letter in ['A', 'B']:
  for i in range(1, 101):
    filename = "%s/%s%d.jpg" % (letter, letter, i)
    im = Image.open(filename)
    X_train.append(list(itertools.chain(*list(im.getdata()))))
    y_train.append(letter)
  for i in range(2991, 3001):
    filename = "%s/%s%d.jpg" % (letter, letter, i)
    im = Image.open(filename)
    X_test.append(list(itertools.chain(*list(im.getdata()))))
    y_test.append(letter)

param_grid = dict(C=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
clf = model_selection.GridSearchCV(linear_model.LogisticRegression(),
                                   param_grid, cv=5)
clf.fit(X_train, y_train)
print clf.best_params_
y_pred = list(clf.predict(X_test))
print metrics.classification_report(y_test, y_pred)
print y_test
print y_pred
