
%matplotlib inline
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import sklearn
import h5py
from sklearn import metrics
import pickle
import numpy as np
import h5py
from numpy.random import permutation
%matplotlib inline
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
import pandas as pd
import numpy as np
import h5py
import csv 
import networkx as nx
from sklearn.metrics import f1_score
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis





filenames = ["abalone19",  
"led7digit-0-2-4-5-6-7-8-9_vs_1",
"abalone9-18",				     
"magic",
"banana",			     
"bands",				     
"mammographic",
"bupa",				     
"monk-2",
"cleveland-0_vs_4",			     
"new-thyroid1",
"coil2000",				     
"new-thyroid2",
"data_banknote_authentication",	     
"page-blocks0",
"ecoli-0_vs_1",			     
"page-blocks-1-3_vs_4",
"ecoli-0-1_vs_2-3-5",			     
"phoneme",
"ecoli-0-1_vs_5",			     
"pima",
"ecoli-0-1-3-7_vs_2-6",		     
"ecoli-0-1-4-6_vs_5",			     
"ring",
"ecoli-0-1-4-7_vs_2-3-5-6",		     
"saheart",
"ecoli-0-1-4-7_vs_5-6",		     
"ecoli-0-2-3-4_vs_5",			     
"segment0",
"ecoli-0-2-6-7_vs_3-5",		     
"ecoli-0-3-4_vs_5",			     
"shuttle-c0-vs-c4",
"ecoli-0-3-4-6_vs_5",			     
"shuttle-c2-vs-c4",
"ecoli-0-3-4-7_vs_5-6",		     
"sonar",
"ecoli-0-4-6_vs_5",			     
"spambase",
"ecoli-0-6-7_vs_3-5",			     
"spectfheart",
"ecoli-0-6-7_vs_5",			     
"titanic",
"ecoli1",				     
"transfusion",
"ecoli2",				     
"twonorm",
"ecoli3",				     
"vehicle0",
"ecoli4",				     
"vehicle1",
"vehicle2",
"vehicle3",
"glass0",				     
"vowel0",
"glass-0-1-2-3_vs_4-5-6",		     
"glass-0-1-4-6_vs_2",			     
"waveform",
"glass-0-1-5_vs_2",			     
"wdbc",
"glass-0-1-6_vs_2",			     
"wisconsin",
"glass-0-1-6_vs_5",			     
"yeast-0-2-5-6_vs_3-7-8-9",
"glass-0-4_vs_5",			     
"yeast-0-2-5-7-9_vs_3-6-8",
"glass-0-6_vs_5",			     
"yeast-0-3-5-9_vs_7-8",
"glass1",				     
"yeast-0-5-6-7-9_vs_4",
"glass2",				     
"yeast1",
"glass4",				     
"yeast-1_vs_7",
"glass5",				     
"yeast-1-2-8-9_vs_7",
"glass6",				     
"yeast-1-4-5-8_vs_7",
"haberman",				     
"yeast-2_vs_4",
"heart",				     
"yeast-2_vs_8",
"hepatitis",				     
"yeast3",
"yeast4",
"ionosphere",				     
"yeast5",
"iris0",				     
"yeast6",
"Indian_Liver_Patient_(ILPD)"] 

# filenames = ['hayes-roth']

number_of_columns = []
for t in range(len(filenames)):
    if( filenames[t] == "Indian_Liver_Patient_(ILPD)" ):
        data = pd.read_csv("E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CIL Code\\ECML-PKDD\\Real_data\\" + filenames[t] + ".csv", header=None)#,names=list_for_names_of_columns)
    else:
        data = pd.read_csv("E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CIL Code\\ECML-PKDD\\Real_data\\" + filenames[t] + ".dat",sep=',', header=None)#,names=list_for_names_of_columns)


    data.head()
    r,c = data.shape
    
    number_of_columns.append(c)
    
delete_datasets = []
for i in range(len(number_of_columns)):
    if(number_of_columns[i] > 30):
        delete_datasets.append(i)
        
        
# filenames = ["abalone9-18", "appendicitis",
   
#              "banana","bands","bupa","coil2000","ecoli-0_vs_1","ecoli1","ecoli2","ecoli3","glass-0-1-2-3_vs_4-5-6","glass0","glass1","glass6","haberman","heart","hepatitis","ionosphere","iris0","magic","mammographic","monk-2","new-thyroid1","newthyroid2","page-blocks0","phoneme","pima","ring","saheart","segment0","sonar","spambase","spectfheart","titanic","twonorm","vehicle0","vehicle1","vehicle2","vehicle3","wdbc","wisconsin","yeast1","yeast3"]

change = ["shuttle-c2-vs-c4","transfusion","vowel0",'bands','bupa','glass0','heart','iris0','mammographic','pima','ring','saheart','sonar','spambase','twonorm','vehicle0','wdbc']
change_new = []
number_of_columns = []
real_datasets = []
for t in range(len(filenames)):
    print("################################################################################################################")
    print("################################################################################################################")

    
#     if(t in delete_datasets):
#         continue
    print(filenames[t])
    
    if( filenames[t] == "Indian_Liver_Patient_(ILPD)" ):
        data = pd.read_csv("E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CIL Code\\ECML-PKDD\\Real_data\\" + filenames[t] + ".csv", header=None)#,names=list_for_names_of_columns)
    else:
        data = pd.read_csv("E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CIL Code\\ECML-PKDD\\Real_data\\" + filenames[t] + ".dat",sep=',', header=None)#,names=list_for_names_of_columns)


#     data.head()
    r,c = data.shape
    
    number_of_columns.append(c)
    data = np.array(data)


    for k in range(len(data)):
        data[k,c-1] = str(data[k,c-1]).strip()
    
    if(filenames[t] == "abalone19" or filenames[t] == "abalone9-18"):
        data[k,0] = str(data[k,0]).strip()
        data = pd.DataFrame(data=data)

        col=[0]#categorical colums
        tmp=[];
        for i,j in enumerate(col):

            a=data[j].unique()
            tmp.append(dict(zip(a,list(range(0,len(a))))))



            data[j]=data[j].map(tmp[i])

        
    if(filenames[t] == "saheart"): 
        data[k,4] = str(data[k,4]).strip()
        data = pd.DataFrame(data=data)

        col=[4]#categorical colums
        tmp=[];
        for i,j in enumerate(col):

            a=data[j].unique()
            tmp.append(dict(zip(a,list(range(0,len(a))))))


            data[j]=data[j].map(tmp[i])

        
    if(filenames[t] == "Indian_Liver_Patient_(ILPD)"): 
        data[k,1] = str(data[k,1]).strip()
        data = pd.DataFrame(data=data)

        col=[1]#categorical colums
        tmp=[];
        for i,j in enumerate(col):

            a=data[j].unique()
            tmp.append(dict(zip(a,list(range(0,len(a))))))


            data[j]=data[j].map(tmp[i])

    
    
     
        
    data = pd.DataFrame(data=data)

    col=[c-1]#categorical colums
    tmp=[];
    for i,j in enumerate(col):

        print(data)
        a=data[j].unique()
        tmp.append(dict(zip(a,list(range(0,len(a))))))

        if(filenames[t] in change):
            temp = tmp[i][a[0]]
            tmp[i][a[0]] = tmp[i][a[1]]
            tmp[i][a[1]] = temp
            
        if(filenames[t] == "waveform"):
            tmp[i][a[2]] = tmp[i][a[0]]
            
    
            
        data[j]=data[j].map(tmp[i])
    print("Class distribution for dataset: " + filenames[t])
    count = pd.value_counts(data[c-1].values, sort=False)
    print("count" , count)
    if(count[0] < count[1]):
        change_new.append(filenames[t])
        
        
#     print(np.array(data))
    real_datasets.append(np.array(data))
        



print(np.shape(real_datasets[0][:,0:2]))
print(np.shape(real_datasets[0][:,2]))
np.save("E:\\Internships_19\\Internship(Summer_19)\\Imbalanced_class_classification\\Class_Imabalanced_Learning_Code\\CIL Code\\ECML-PKDD\\Real_data\\real_datasets_final_done.npy", real_datasets)


for j in range(len(real_datasets)):
    print("dataset: ", j)
#     for i in range(len(real_datasets[j])):
    print(np.shape(real_datasets[j]))

# 
