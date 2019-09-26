from sklearn import svm
import numpy as np
import cv2
from autoCanny import auto_canny
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train =[]
label =[]
for i in range(0,10):
    for j in range(1,3795):
        path = "hog/" + str(i) + "/" + "img (" + str(j) + ").jpg"
        img = cv2.imread(path)
        img = cv2.resize(img, (40,40))
        cv2.imshow(str(i)+"img",img)
        img = img.reshape(-1,1)[0,:]
        train.append(img)
        label.append(i)

train=np.array(train)
X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.2)
# svm canny edge
clf = svm.SVC(gamma='scale',kernel='rbf')
clf.fit(X_train,y_train)
print(clf.get_params())

y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("svm dataset")
print(score)
# print(clf.score(X_test,y_test))

clf = svm.SVC(gamma='scale',kernel='rbf')
clf.fit(X_train,y_train)
print(clf.get_params())

y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)


print(score)


#cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, train, label, cv=2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


cv2.waitKey()
cv2.destroyAllWindows()

