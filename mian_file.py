from create_data_file import *
from sklearn.metrics import confusion_matrix
from SpliceAI_prostate.train_new import SpliceAImodel
from CNN import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from metrics import model_result

np.random.seed(123)
data = read_txt_file(r"./HeLa_train_data/total_x.txt")
data = transform_x_one_hot_set(data)
y = read_txt_file(r"./HeLa_train_data/total_y.txt")
Y = transform_Y_one_hot_set(y, 1)
Y = np.array(Y)
length=Y.shape[0]
shuffle_list=np.random.permutation(length)
Y=Y[shuffle_list,:]
data=data[shuffle_list,:,:]


# create sub set to remove some type 1 data
y_array = np.array(y)
y_sub = np.concatenate([y_array[y_array != 1], y_array[y_array == 1][0:1800]], axis=0)
y_sub = y_sub.reshape((-1, 1))
Y_sub = np.array(transform_Y_one_hot_set(y_sub, 1))
data_idx = np.concatenate([np.reshape(y_array != 1, (-1)), np.reshape(y_array == 1, (-1))[0:1800]])
data_sub = np.concatenate([data[np.reshape(y_array != 1, (-1))], data[np.reshape(y_array == 1, (-1))][0:1800]], axis=0)

#train test split
train_x,test_x,train_y,test_y=train_test_split(data_sub,Y_sub,test_size=0.15,random_state=123)
'''
t=1: simple CNN model
t=2:spliceAI model
t=3 our designed CNN model
t=4 our designed CNN model(with dropout layer)
'''
t=3

if(t==1):
    CNN=CNN_model()

    CNN.fit(train_x,train_y,validation_data=(test_x,test_y),epochs=20,batch_size=32,shuffle=True)
    y_pre=CNN.predict(data)
    y_pre_label=[np.argmax(one_hot)for one_hot in y_pre]
    y_ture_label=[np.argmax(one_hot)for one_hot in Y]
    print(confusion_matrix(y_pre_label,y_ture_label))

    y_pre2=CNN.predict(test_x)
    y_pre_label2=[np.argmax(one_hot)for one_hot in y_pre2]
    y_ture_label2=[np.argmax(one_hot)for one_hot in test_y]
    print(roc_auc_score(test_y,y_pre2))
    print(confusion_matrix(y_pre_label2,y_ture_label2))


#ResNet

if(t==2):

    y_pre2,test_y_clip=SpliceAImodel(train_x,train_y,test_x,test_y,80)
    model_result(test_y_clip,y_pre2,"spliceai_type1_1800_HeLa1")
if(t==3):

    ownModel=ownModel()
    #ownModel.fit([train_x[:,0:300,:],train_x[:,250:550,:],train_x[:,350:650,:],train_x[:,550:850,:],train_x[:,650:950,:],train_x[:,900:1200,:]], train_y,
     #            validation_data=([test_x[:,0:300,:],test_x[:,250:550,:],test_x[:,350:650,:],test_x[:,550:850,:],test_x[:,650:950,:],test_x[:,900:1200,:]],test_y),
      #           epochs = 20, batch_size = 32, shuffle = True)
    ownModel.fit([train_x[:,0:300,:],train_x[:,200:500,:],train_x[:,300:600,:],train_x[:,600:900,:],train_x[:,700:1000,:],train_x[:,900:1200,:]], train_y,
                 validation_data=([test_x[:,0:300,:],test_x[:,200:500,:],test_x[:,300:600,:],test_x[:,600:900,:],test_x[:,700:1000,:],test_x[:,900:1200,:]],test_y),
                 epochs = 10, batch_size = 16, shuffle = True)
    #ownModel.fit([train_x[:,0:200,:],train_x[:,200:400,:],train_x[:,400:600,:],train_x[:,600:800,:],train_x[:,800:1000,:],train_x[:,1000:1200,:]], train_y,
     #            validation_data=([test_x[:,0:200,:],test_x[:,200:400,:],test_x[:,400:600,:],test_x[:,600:800,:],test_x[:,800:1000,:],test_x[:,1000:1200,:]],test_y),
      #           epochs = 20, batch_size = 32, shuffle = True)

    #y_pre=ownModel.predict([data[:,0:300,:],data[:,250:550,:],data[:,350:650,:],data[:,550:850,:],data[:,650:950,:],data[:,900:1200,:]])
    y_pre=ownModel.predict([data[:,0:300,:],data[:,200:500,:],data[:,300:600,:],data[:,600:900,:],data[:,700:1000,:],data[:,900:1200,:]])
    y_pre_label=[np.argmax(one_hot)for one_hot in y_pre]
    y_ture_label=[np.argmax(one_hot)for one_hot in Y]
    print(confusion_matrix(y_pre_label,y_ture_label))
    print("-----------------------------------------")
   # y_pre2 = ownModel.predict(
    #    [test_x[:, 0:300, :], test_x[:, 250:550, :], test_x[:, 350:650, :], test_x[:, 550:850, :], test_x[:, 650:950, :],
     #    test_x[:, 900:1200, :]])
    #y_pre2 = ownModel.predict(
     #   [test_x[:, 0:200, :], test_x[:, 200:400, :], test_x[:, 400:600, :], test_x[:, 600:800, :], test_x[:, 800:1000, :],
      #   test_x[:, 1000:1200, :]])
    y_pre2 = ownModel.predict(
        [test_x[:,0:300,:],test_x[:,200:500,:],test_x[:,300:600,:],test_x[:,600:900,:],test_x[:,700:1000,:],test_x[:,900:1200,:]])

    model_result(test_y,y_pre2,"ownModel1_test0.15_ce_adadelta_epoch10_16_type1_1800_Hela3")


if(t==4):

    ownModel=ownModel2()
    #ownModel.fit([train_x[:,0:300,:],train_x[:,250:550,:],train_x[:,350:650,:],train_x[:,550:850,:],train_x[:,650:950,:],train_x[:,900:1200,:]], train_y,
     #            validation_data=([test_x[:,0:300,:],test_x[:,250:550,:],test_x[:,350:650,:],test_x[:,550:850,:],test_x[:,650:950,:],test_x[:,900:1200,:]],test_y),
      #           epochs = 20, batch_size = 32, shuffle = True)
    ownModel.fit([train_x[:,0:300,:],train_x[:,200:500,:],train_x[:,300:600,:],train_x[:,600:900,:],train_x[:,700:1000,:],train_x[:,900:1200,:]], train_y,
                 validation_data=([test_x[:,0:300,:],test_x[:,200:500,:],test_x[:,300:600,:],test_x[:,600:900,:],test_x[:,700:1000,:],test_x[:,900:1200,:]],test_y),
                 epochs = 10, batch_size = 16, shuffle = True)
    #ownModel.fit([train_x[:,0:200,:],train_x[:,200:400,:],train_x[:,400:600,:],train_x[:,600:800,:],train_x[:,800:1000,:],train_x[:,1000:1200,:]], train_y,
     #            validation_data=([test_x[:,0:200,:],test_x[:,200:400,:],test_x[:,400:600,:],test_x[:,600:800,:],test_x[:,800:1000,:],test_x[:,1000:1200,:]],test_y),
      #           epochs = 20, batch_size = 32, shuffle = True)

    #y_pre=ownModel.predict([data[:,0:300,:],data[:,250:550,:],data[:,350:650,:],data[:,550:850,:],data[:,650:950,:],data[:,900:1200,:]])
    y_pre=ownModel.predict([data[:,0:300,:],data[:,200:500,:],data[:,300:600,:],data[:,600:900,:],data[:,700:1000,:],data[:,900:1200,:]])
    y_pre_label=[np.argmax(one_hot)for one_hot in y_pre]
    y_ture_label=[np.argmax(one_hot)for one_hot in Y]
    print(confusion_matrix(y_pre_label,y_ture_label))
    print("-----------------------------------------")
   # y_pre2 = ownModel.predict(
    #    [test_x[:, 0:300, :], test_x[:, 250:550, :], test_x[:, 350:650, :], test_x[:, 550:850, :], test_x[:, 650:950, :],
     #    test_x[:, 900:1200, :]])
    #y_pre2 = ownModel.predict(
     #   [test_x[:, 0:200, :], test_x[:, 200:400, :], test_x[:, 400:600, :], test_x[:, 600:800, :], test_x[:, 800:1000, :],
      #   test_x[:, 1000:1200, :]])
    y_pre2 = ownModel.predict(
        [test_x[:,0:300,:],test_x[:,200:500,:],test_x[:,300:600,:],test_x[:,600:900,:],test_x[:,700:1000,:],test_x[:,900:1200,:]])

    model_result(test_y,y_pre2,"ownModel1_test0.15_ce_adadelta_epoch10_16_type1_1800_Hela3")