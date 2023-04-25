#!/usr/bin/env python
# coding: utf-8

# In[493]:


import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
enc=LabelEncoder()
from sklearn.metrics import roc_curve, auc 
import matplotlib.pyplot as plt


# # credit card fraud dataset https://www.kaggle.com/datasets/jacklizhi/creditcard

# In[494]:


df=pd.read_csv(r"C:\Users\AMIT VASHISTHA\Downloads\archive (3)\creditcard.csv")
df.columns = [*df.columns[:-1], 'Class']
df.loc[:,['Class']]=df.loc[:,['Class']].apply(enc.fit_transform)
print(df.head(20))


# In[495]:



print(df['Class'].value_counts())


# In[496]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[497]:


fl=df.to_numpy()
x=fl[:,:-1]
y=fl[:,-1]


# In[498]:


trainX, testX_, trainy, testy_ = train_test_split(x, y, test_size=0.3,random_state=10)


# In[499]:


model = LogisticRegression()
model.fit(trainX, trainy)
y_pred=model.predict(testX_)
acc=accuracy_score(testy_,y_pred)


# In[500]:


from sklearn import metrics
import matplotlib.pyplot as plt
confusion_matrix = metrics.confusion_matrix(testy_,y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
from sklearn.metrics import accuracy_score
print(accuracy_score(testy_,y_pred))
plt.show()
confusion_matrix=confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
print(confusion_matrix.diagonal())


# In[501]:


print(acc)


# In[502]:


def under_sampling(data,display_matrix='no'):
    train, test = train_test_split(data,test_size=0.3,random_state=10)
    a,b=train.Class.value_counts()
    
    if a<b:
        minority=0
        N=a
    else:
        minority=1
        N=b
    
    fl1=train[train['Class']==minority]
    fl2=train[train['Class']==1-minority].sample(n=N)
    
    
    
    fl_=fl1.append(fl2)
    fl_ = fl_.sample(frac=1).reset_index(drop=True)
    
    fl_=fl_.to_numpy()
    trainx=fl_[:,:-1]
    trainy=fl_[:,-1]
    
    print(trainx.shape)
    
    test=test.to_numpy()
    testx=test[:,:-1]
    testy=test[:,-1]
    
    #trainX, testX_, trainy, testy_ = train_test_split(x, y, test_size=0.3, random_state=2)
    model = LogisticRegression()
    model.fit(trainx, trainy)
    y_pred=model.predict(testx)
    lr_probs  = model.predict_proba(testx)
    lr_probs = lr_probs[:, 1]
    ns_probs = [0 for _ in range(len(testy))]

    svm_fpr, svm_tpr, threshold = roc_curve(testy, lr_probs)
    #ns_fpr, ns_tpr, _ = roc_curve(testy_, ns_probs)
    auc_svm = auc(svm_fpr, svm_tpr)
    acc=accuracy_score(testy,y_pred)
    
    if display_matrix=='yes':
        confusion_matrix = metrics.confusion_matrix(testy,y_pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        cm_display.plot()
        plt.show()
        confusion_matrix=confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print(confusion_matrix.diagonal())
    
    return svm_fpr, svm_tpr,auc_svm,acc
    


# In[503]:


svm_fpr, svm_tpr,auc_svm,acc_us=under_sampling(df,'yes')


# In[504]:


print(acc_us)


# In[ ]:





# In[505]:


from sklearn.neighbors import NearestNeighbors
class SMOTE():
    

    def __init__(self, k_neighbors=5, random_state=None):
        self.k = k_neighbors
        self.random_state = random_state

    def sample(self, n_samples):
        
        np.random.seed(seed=self.random_state)

        S = np.zeros(shape=(n_samples, self.n_features))
        # Calculate synthetic samples.
        for i in range(n_samples):
            j = np.random.randint(0, self.X.shape[0])

            # Find the NN for each sample.
            # Exclude the sample itself.
            nn = self.neigh.kneighbors(self.X[j].reshape(1, -1), return_distance=False)[:, 1:]
            
            nn_index = np.random.choice(nn[0])
            #print('random',nn_index)

            dif = self.X[nn_index] - self.X[j]
            gap = np.random.random()

            S[i, :] = self.X[j, :] + gap * dif[:]
        S=np.concatenate((self.X, S), axis=0)
        return S

    def fit(self, X):
        
        self.X = X
        self.n_minority_samples, self.n_features = self.X.shape

        # Learn nearest neighbors.
        self.neigh = NearestNeighbors(n_neighbors=self.k + 1)
        self.neigh.fit(self.X)

        return self


# In[506]:


def smote_l(data,display_matrix='no'):
    
    train, test = train_test_split(data,test_size=0.3,random_state=10)
    a,b=train.Class.value_counts()
    
    if a<b:
        minority=0
        N=a
    else:
        minority=1
        N=b
    over_sample=abs(a-b)
    df2=train[train['Class']==minority]    #minority class
    df1=train[train['Class']==1-minority] 
    df2=df2.to_numpy()
    
    


    sm1=SMOTE()
    sm1.fit(df2)
    smp=sm1.sample(int(over_sample))
    
    
    df1=df1.to_numpy()
    df_=np.concatenate((df1, smp), axis=0)
    np.random.shuffle(df_) 
    
    trainX=df_[:,:-1]
    trainy=df_[:,-1]
    trainy=np.ravel(trainy)
    
    test=test.to_numpy()
    testx=test[:,:-1]
    testy=test[:,-1]

    model = LogisticRegression()
    model.fit(trainX, trainy)
    y_pred=model.predict(testx)
    lr_probs  = model.predict_proba(testx)
    lr_probs = lr_probs[:, 1]
    ns_probs = [0 for _ in range(len(testy))]

    svm_fpr, svm_tpr, threshold = roc_curve(testy, lr_probs)
    #ns_fpr, ns_tpr, _ = roc_curve(testy_, ns_probs)
    auc_svm = auc(svm_fpr, svm_tpr)
    acc=accuracy_score(testy,y_pred)
    if display_matrix=='yes':
        confusion_matrix = metrics.confusion_matrix(testy,y_pred)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
        cm_display.plot()
        plt.show()
        confusion_matrix=confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print(confusion_matrix.diagonal())
    return svm_fpr, svm_tpr,auc_svm,acc


# In[507]:


smt_svm_fpr, smt_svm_tpr,smt_auc_svm,acc_s=smote_l(df,'yes')


# In[508]:


print(acc_s)


# In[509]:


plt.plot(smt_svm_fpr, smt_svm_tpr, linestyle='-', label='(under sampling auc = %0.3f)' % smt_auc_svm)
plt.plot(svm_fpr, svm_tpr, linestyle='-', label='(smote sample auc = %0.3f)' % auc_svm)
#plt.plot(ns_fpr, ns_tpr,linestyle='-')


plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




