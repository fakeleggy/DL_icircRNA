import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

def roc_plot_function(test_y,y_pre2,title):
    n_classes = test_y.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_pre2[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), y_pre2.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw=2
    plt.figure(figsize=(9,9))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})' 
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})' 
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'lightgreen', 'orchid'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})' 
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right",prop={'size':8})
    plt.savefig('model_result/'+title+'/ROC_AUC_plot.png')
    plt.show()

def confusion_Matrix(test_y,y_pre2,title):
    y_pre_label2 = [np.argmax(one_hot) for one_hot in y_pre2]
    y_ture_label2 = [np.argmax(one_hot) for one_hot in test_y]
    df=pd.DataFrame(confusion_matrix(y_pre_label2, y_ture_label2))
    df.to_excel("model_result/"+title+"/confusion_matrix.xlsx",header=False,index=False)
    print(df)


def model_result(test_y,y_pre2,title):
    if not os.path.isdir('model_result/' + title):
        os.makedirs('model_result/'+title)
    confusion_Matrix(test_y,y_pre2,title)
    roc_plot_function(test_y,y_pre2,title)
