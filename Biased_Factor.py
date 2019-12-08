# coding: utf-8

# ## Biased or SlantedNews detection Factor
# #### This notebook is a subset of Biased_Factor.ipynb

# In[1]:


import numpy as np
import pandas as pd
import warnings
import pickle

#labelcolname = 'Encoded_Label'
#titlecolname = 'Speaker\'sJobTitle'
#true_labels = ['original','true','mostly-true','half-true']
#false_labels = ['barely-true','false','pants-fire']      

def prediction(xtest): 
    xtest = xtest.split(' ')
    pickleModel = "/content/gdrive/My Drive/Drifters/Models/Biased_Model.pkl"
    pickle_in = open(pickleModel, "rb")
    loadData = pickle.load(pickle_in)   
    dataset = loadData.predict(xtest)
    for i in dataset:
        if(i == 0):
            return 0
    return 1
            

#def simplify_label(input_label):
#    if input_label in true_labels:
#        return 1
#    else:
#        return 0
    
class BiasedDetection:
    def __init__(self, xtest):
        self.x_test = xtest
        #self.y_test = xtest.apply(lambda row: simplify_label(row['Label']), axis=1)
        #self.x_test = self.x_test.replace(np.nan,'No Job Title', regex=True)
    def predict(self):
        return prediction(self.x_test)

