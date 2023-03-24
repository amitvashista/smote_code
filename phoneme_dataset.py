import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()


df=pd.read_csv(r"C:\Users\AMIT VASHISTHA\Downloads\phoneme_csv.csv")
df.loc[:,['Class']]=df.loc[:,['Class']].apply(enc.fit_transform)
print(df.Class.value_counts())
df1=df[df['Class']==0].sample(n=1585)
df2=df[df['Class']==1].sample(n=1585)
df_=df1.append(df2)
df_ = df_.sample(frac=1).reset_index(drop=True)

df_=df_.to_numpy()
x=df_[:,:5]
y=df_[:,5:6]
y=np.ravel(y)

trainX, testX_, trainy, testy_ = train_test_split(x, y, test_size=0.3, random_state=2)
model = LogisticRegression()
model.fit(trainX, trainy)
y_pred=model.predict(testX_)
lr_probs  = model.predict_proba(testX_)
lr_probs = lr_probs[:, 1]
ns_probs = [0 for _ in range(len(testy_))]

svm_fpr, svm_tpr, threshold = roc_curve(testy_, lr_probs)
ns_fpr, ns_tpr, _ = roc_curve(testy_, ns_probs)
auc_svm = auc(svm_fpr, svm_tpr)

#plt.figure(figsize=(5, 5), dpi=100)
plt.plot(svm_fpr, svm_tpr, linestyle='-', label='(under sample auc = %0.3f)' % auc_svm)


plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

plt.legend()


from collections import Counter
import pandas as pd
import numpy as np
from sklearn.base import is_regressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble._forest import BaseForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.tree import BaseDecisionTree
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
#from sklearn.utils import shuffle

class SMOTE(object):
    """Implementation of Synthetic Minority Over-Sampling Technique (SMOTE).
    SMOTE performs oversampling of the minority class by picking target 
    minority class samples and their nearest minority class neighbors and 
    generating new samples that linearly combine features of each target 
    sample with features of its selected minority class neighbors [1].
    Parameters
    ----------
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.
    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O. Hall, and P. Kegelmeyer. "SMOTE:
           Synthetic Minority Over-Sampling Technique." Journal of Artificial
           Intelligence Research (JAIR), 2002.
    """

    def __init__(self, k_neighbors=5, random_state=None):
        self.k = k_neighbors
        self.random_state = random_state

    def sample(self, n_samples):
        """Generate samples.
        Parameters
        ----------
        n_samples : int
            Number of new synthetic samples.
        Returns
        -------
        S : array, shape = [n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)

        S = np.zeros(shape=(n_samples, self.n_features))
        # Calculate synthetic samples.
        for i in range(n_samples):
            j = np.random.randint(0, self.X.shape[0])

            # Find the NN for each sample.
            # Exclude the sample itself.
            nn = self.neigh.kneighbors(self.X[j].reshape(1, -1), return_distance=False)[:, 1:]
            nn_index = np.random.choice(nn[0])

            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]

        return S

    def fit(self, X):
        """Train model based on input data.
        Parameters
        ----------
        X : array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples.
        """
        self.X = X
        self.n_minority_samples, self.n_features = self.X.shape

        # Learn nearest neighbors.
        self.neigh = NearestNeighbors(n_neighbors=self.k + 1)
        self.neigh.fit(self.X)

        return self


df2=df2.to_numpy()
sm1=SMOTE()
sm1.fit(df2)
smp=sm1.sample(2232)

df_=df.to_numpy()
df_=np.concatenate((df, smp), axis=0)
np.random.shuffle(df_) 


x=df_[:,:5]
y=df_[:,5:6]
y=np.ravel(y)

trainX, testX_, trainy, testy_ = train_test_split(x, y, test_size=0.3, random_state=2)
model = LogisticRegression()
model.fit(trainX, trainy)
y_pred=model.predict(testX_)
lr_probs  = model.predict_proba(testX_)
lr_probs = lr_probs[:, 1]
ns_probs = [0 for _ in range(len(testy_))]

svm_fpr, svm_tpr, threshold = roc_curve(testy_, lr_probs)
ns_fpr, ns_tpr, _ = roc_curve(testy_, ns_probs)
auc_svm = auc(svm_fpr, svm_tpr)

#plt.figure(figsize=(5, 5), dpi=100)
plt.plot(svm_fpr, svm_tpr, linestyle='-', label='(smote sample auc = %0.3f)' % auc_svm)
plt.plot(ns_fpr, ns_tpr,linestyle='-')


plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

plt.legend()
plt.show()


'''df = pd.DataFrame(df, columns = ['V1',	'V2',	'V3',	'V4',	'V5',	'Class'])
print(df.Class.value_counts())'''
