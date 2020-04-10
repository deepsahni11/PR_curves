import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
import numpy as np
import h5py
import pickle
import sklearn
import random 
from sklearn.metrics import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import *
from imblearn.ensemble import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification
from scipy import stats
from tqdm import *
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from imblearn.metrics import geometric_mean_score
from numpy.random import permutation
from sklearn.metrics import f1_score
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

seed = 0
samplers_all = [
    # Oversampling methods:
    RandomOverSampler(random_state=seed), 
    SMOTE(random_state=seed),             
    ADASYN(random_state=seed),            
    BorderlineSMOTE(random_state=seed),
    SVMSMOTE(random_state=seed),
    
    # Undersampling methods:
    RandomUnderSampler(random_state=seed),
    ClusterCentroids(random_state=seed),
    NearMiss(version=1, random_state=seed),
    NearMiss(version=2, random_state=seed),
    NearMiss(version=3, random_state=seed),
    TomekLinks(random_state=seed),
    EditedNearestNeighbours(random_state=seed),
    RepeatedEditedNearestNeighbours(random_state=seed),
    AllKNN(random_state=seed),
    CondensedNearestNeighbour(random_state=seed),
    OneSidedSelection(random_state=seed),
    NeighbourhoodCleaningRule(random_state=seed),
    InstanceHardnessThreshold(random_state=seed),
    
    
    # Combos:
    SMOTEENN(random_state=seed),
    SMOTETomek(random_state=seed)

]
samplers_array_all = np.array(samplers_all)

samplerAbbrev = [
    "ROS",
    "SMOTE",
    "ADASYN",
    "B-SMOTE",
    "SVMSMOTE",
    "RUS",
    "CC",
    "NM-1",
    "NM-2",
    "NM-3",
    "Tomek",
    "ENN",
    "RENN",
    "AkNN",
    "CNN",
    "OSS",
    "NCR",
    "IHT",
    "SMOTE+ENN",
    "SMOTE+Tomek"
]

seed = 0

%matplotlib inline

def draw_cv_roc_curve(classifier, cv, X, y, title='ROC Curve'):
    """
    Draw a Cross Validated ROC Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series
    Example largely taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    # Creating ROC Curve with Cross Validation
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def draw_cv_pr_curve(classifier, cv, Xtrain, ytrain, Xtest, ytest, label,title='PR Curve'):
    """
    Draw a Cross Validated PR Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series

    Largely taken from: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
    """
    y_real = []
    y_proba = []

#     i = 0
#     for train, test in cv.split(X, y):
    probas_ = classifier.fit(Xtrain, ytrain).predict_proba(Xtest)
    # Compute ROC curve and area the curve
    precision, recall, _ = precision_recall_curve(ytest, probas_[:, 1])
#     plt.figure(figsize=(20,10))

    # Plotting each individual PR Curve
#         plt.plot(recall, precision, lw=1, alpha=0.3,
#                  label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y.iloc[test], probas_[:, 1])))

    y_real.append(ytest)
    y_proba.append(probas_[:, 1])

#         i += 1

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    
#     plt.figure(figsize=(20,10))
    plt.plot(recall, precision, label=label,lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
#     plt.show()
   


    
# Create a fake example where X is an 1000 x 2 Matrix
# Y is 1000 x 1 vector
# Binary Classification Problem
#FOLDS = 5

# X, y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=10.0,
#     random_state=12345)

#X, y = make_classification(n_samples = 1000,n_features=5, n_redundant=0, n_informative=5,
#                             n_clusters_per_class=3,  flip_y=0.08,random_state=3)



for r in range(len(real_
    X = np.array(real_datasets[r][:,:-1])
    y = np.array(real_datasets[r][:,-1])
    y=y.astype('int')


    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    # f, axes = plt.subplots(1, 2, figsize=(10, 5))

    # X.loc[y.iloc[:, 0] == 1]

    # axes[0].scatter(X.loc[y.iloc[:, 0] == 0, 0], X.loc[y.iloc[:, 0] == 0, 1], color='blue', s=2, label='y=0')
    # axes[0].scatter(X.loc[y.iloc[:, 0] !=0, 0], X.loc[y.iloc[:, 0] != 0, 1], color='red', s=2, label='y=1')
    # axes[0].set_xlabel('X[:,0]')
    # axes[0].set_ylabel('X[:,1]')
    # axes[0].legend(loc='lower left', fontsize='small')


    # classifier = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=0)
    # Setting up simple RF Classifier
    clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(10, 5), random_state=0)

    # Set up Stratified K Fold
    cv = StratifiedKFold(n_splits=10)


    # draw_cv_roc_curve(clf, cv, X, y, title='Cross Validated ROC')
    # draw_cv_pr_curve(clf, cv, X, y, title='Cross Validated PR Curve')

    plt.figure(figsize=(20,10))





    c = 0
    throw = 0
    Throw = []


    y_real = []
    y_proba = []
    for train, test in cv.split(X, y):

        c = c + 1
        Xtrain, Xtest = X.iloc[train], X.iloc[test]

    #     print(np.shape(Xtrain))
    #     print(np.shape(Xtrain[0]))
        ytrain, ytest = y.iloc[train], y.iloc[test]
    #     X_train_old, y_train_old = X_train, y_train
    #     y_real = []
    #     y_proba = []

    #     i = 0
    #     for train, test in cv.split(X, y):
        probas_ = classifier.fit(Xtrain, ytrain).predict_proba(Xtest)
        # Compute ROC curve and area the curve
        precision, recall, _ = precision_recall_curve(ytest, probas_[:, 1])
    #     plt.figure(figsize=(20,10))

        # Plotting each individual PR Curve
    #         plt.plot(recall, precision, lw=1, alpha=0.3,
    #                  label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y.iloc[test], probas_[:, 1])))

        y_real.append(ytest)
        y_proba.append(probas_[:, 1])

    #         i += 1

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)


    #     plt.figure(figsize=(20,10))
    plt.plot(recall, precision, label="None",lw=2, alpha=.8)

    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    #     plt.title(title)
    # plt.legend(loc="lower right")

    #     fig = plt.figure(figsize=(15,10))





    c= 0
    for i in range(len(samplers_array_all)):

        try:
            y_real = []
            y_proba = []


            count = 0
            for train, test in cv.split(X, y):

                count = count + 1
                print(count)
                Xtrain, Xtest = X.iloc[train], X.iloc[test]

            #     print(np.shape(Xtrain))
            #     print(np.shape(Xtrain[0]))
                ytrain, ytest = y.iloc[train], y.iloc[test]

                X_tr, y_tr = samplers_array_all[i].fit_sample(Xtrain, ytrain)
    #             y_real = []
    #             y_proba = []

            #     i = 0
            #     for train, test in cv.split(X, y):
                probas_ = classifier.fit(X_tr, y_tr).predict_proba(Xtest)
                # Compute ROC curve and area the curve
                precision, recall, _ = precision_recall_curve(ytest, probas_[:, 1])
            #     plt.figure(figsize=(20,10))

                # Plotting each individual PR Curve
            #         plt.plot(recall, precision, lw=1, alpha=0.3,
            #                  label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y.iloc[test], probas_[:, 1])))

                y_real.append(ytest)
                y_proba.append(probas_[:, 1])

            #         i += 1

            y_real = np.concatenate(y_real)
            y_proba = np.concatenate(y_proba)

            precision, recall, _ = precision_recall_curve(y_real, y_proba)


        #     plt.figure(figsize=(20,10))
            plt.plot(recall, precision, label=str(i+1),lw=2, alpha=.8)


    #             plt.title(title)
    #         plt.legend(loc="lower right")

        except:

            throw = throw + 1
            Throw.append(str(samplers_array_all[i]))
            continue


    plt.legend()
    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig("Precision-Recall_curve_for_real_dataset_" + str(r) + ".pdf" )

    plt.show()
    #     draw_cv_pr_curve(clf, cv, X, y, title='Cross Validated PR Curve')






    #     probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
    #     # Compute ROC curve and area the curve
    #     precision, recall, _ = precision_recall_curve(y.iloc[test], probas_[:, 1])


    #     # Plotting each individual PR Curve
    # #     plt.plot(recall, precision, lw=1, alpha=0.3,
    # #              label='PR fold %d (AUC = %0.2f)' % (i, average_precision_score(y.iloc[test], probas_[:, 1])))

    #     y_real.append(y.iloc[test])
    #     y_proba.append(probas_[:, 1])

    #     i += 1

    # y_real = np.concatenate(y_real)
    # y_proba = np.concatenate(y_proba)

    # precision, recall, _ = precision_recall_curve(y_real, y_proba)


    # # plt.figure(figsize=(20,10))
    # plt.plot(recall, precision, color='b',
    #          label=r'Precision-Recall (AUC = %0.2f)' % (average_precision_score(y_real, y_proba)),
    #          lw=2, alpha=.8)

    # plt.xlim([-0.05, 1.05])
    # plt.ylim([-0.05, 1.05])
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('PR Curve')
    # plt.legend(loc="lower right")
    # plt.show()


