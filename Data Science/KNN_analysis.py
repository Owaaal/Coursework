# Import necessary libraries.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Set up the RNG for numpy.random
RANDOM_SEED = 42
np.random.seed(seed=RANDOM_SEED)
df = pd.read_csv('titanic.csv')

# Create feature Matrix
y = df["Survived"]
X = df.loc[:, ['Sex', 'Age', 'Pclass', 'Siblings/Spouses Aboard', 'Parents/Children Aboard', 'Fare']]
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
#Split into train/test
data_train, data_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2, random_state=42)
unique_label_count = pd.Series(y).nunique()

#This is nearest neighbour? I think
k=5
print('Using k-value of:',k)

try:
    KNN = neighbors.KNeighborsClassifier(
        n_neighbors=k, 
        weights='distance', algorithm='auto', p=1, 
        metric='minkowski')
    KNN.fit(X, y)
    predicted_train = KNN.predict(data_train)
    predicted_test  = KNN.predict(data_test)
    confmat_train = confusion_matrix(labels_train, predicted_train).ravel()
    confmat_test  = confusion_matrix(labels_test, predicted_test).ravel()
except: print('KNN error')

try:
    ACC_train       = (confmat_train[0]+confmat_train[3])/np.sum(confmat_train) # (tp+tn)/(tp+tn+fp+fn)
    precision_train = confmat_train[3]/(confmat_train[1] + confmat_train[3]) # tp/(fp+tp)
    recall_train    = confmat_train[3]/(confmat_train[2] + confmat_train[3]) # tp/(fn+tp)

    ACC_test       = (confmat_test[0]+confmat_test[3])/np.sum(confmat_test)
    precision_test = confmat_test[3]/(confmat_test[1] + confmat_test[3])
    recall_tese    = confmat_test[3]/(confmat_test[2] + confmat_test[3])
except:
    print('training Error')

P_CV = 0.2
MAX_K = 50
data_train_CV, data_val_CV, labels_train_CV, labels_val_CV = train_test_split(data_train, labels_train, test_size=P_CV, random_state=RANDOM_SEED)
ACC_train_CV, ACC_val_CV = [], []
'''
for i in range(1, 50):
    KNN = neighbors.KNeighborsClassifier(
            n_neighbors=i )
    KNN.fit(data_train_CV, labels_train_CV)
    predicted_train = KNN.predict(data_train_CV)
    predicted_val   = KNN.predict(data_val_CV)
    ACC_train   = accuracy_score(labels_train_CV, predicted_train)
    ACC_val     = accuracy_score(labels_val_CV, predicted_val)
    ACC_train_CV.append(ACC_train)
    ACC_val_CV.append(ACC_val)

print('train:',len(ACC_train_CV))
print('test:',len(ACC_val_CV))'''

'''fig_1, ax_1 = plt.subplots()
ax_1.plot(range(1, MAX_K), ACC_train_CV)
ax_1.plot(range(1, MAX_K), ACC_val_CV)
ax_1.set_title('Validation curve')
ax_1.set_xlabel('k number')
ax_1.set_ylabel('classification ACC')
ax_1.legend(['Train','Validation'])
ax_1.grid()
plt.show()'''

RUNS = 10
MAX_K = 50

ACC_train_CV, ACC_val_CV = np.zeros([RUNS, MAX_K-1], dtype=np.float32), np.zeros([RUNS, MAX_K-1], dtype=np.float32)

for r in range(RUNS):
    data_train_CV, data_val_CV, labels_train_CV, labels_val_CV = train_test_split(data_train, labels_train, test_size=P_CV, random_state=r)
    print('starting run:',r)
    for i in range(2, MAX_K+1):
        KNN = neighbors.KNeighborsClassifier(
            n_neighbors=i )
        KNN.fit(data_train_CV, labels_train_CV)

        predicted_train = KNN.predict(data_train_CV)
        predicted_val   = KNN.predict(data_val_CV)
        ACC_train   = accuracy_score(labels_train_CV, predicted_train)
        ACC_val     = accuracy_score(labels_val_CV, predicted_val)
        ACC_train_CV[r, i-2] = ACC_train
        ACC_val_CV[r, i-2] = ACC_val

'''fig_2, ax_2 = plt.subplots()
ax_2.plot(range(2, MAX_K+1), ACC_train_CV.mean(axis=0))
ax_2.plot(range(2, MAX_K+1), ACC_val_CV.mean(axis=0))
ax_2.set_title('Avergaed validation curve')
ax_2.set_xlabel('(k)')
ax_2.set_ylabel('classification ACC')
ax_2.legend(['Train','Validation'])
ax_2.grid()
plt.show()
'''

ind = np.argmax(ACC_val_CV.mean(axis=0))
best_valACC = ACC_val_CV.mean(axis=0)[ind]
k_opt = range(2, MAX_K+1)[ind]

print('------- KNn cross validated with %d time---------------------' % RUNS)
print('Optimal k = %d \n' %  k_opt)
print('Validation_ACC=%e\n' % best_valACC)


print('------- KNN with optimal k found earlier -------')
KNN = neighbors.KNeighborsClassifier(
            n_neighbors=k_opt )
KNN.fit(data_train, labels_train)

predicted_train = KNN.predict(data_train)
predicted_test  = KNN.predict(data_test)
ACC_train   = accuracy_score(labels_train, predicted_train)
ACC_test    = accuracy_score(labels_test, predicted_test)
print('Training data:\n accuray=%.4f\n' % ACC_train)
print('Test data:\n accuray=%.4f\n' % ACC_test)