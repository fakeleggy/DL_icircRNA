from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D,Conv1D,BatchNormalization,Dense,concatenate,Dropout,add
from keras.layers.core import Activation
from keras.layers import Flatten,Reshape


def CNN_model():
    input_dim=(1200,4)
    input=Input(input_dim)
    x=Conv1D(5,1,activation="relu")(input)
    x=Dropout(0.2)(x)
    x=Conv1D(5,4,padding="same",activation="relu")(x)
    x=Dropout(0.3)(x)
    x=Conv1D(5,8,padding="same",activation="relu")(x)
    x=Dropout(0.5)(x)
    # x=Reshape((60,60,1))(x)
    # x=Conv2D(32,3,padding="same",activation="relu")(x)
    # x=Conv2D(16,3,padding="same",activation="relu")(x)
    x=Flatten()(x)
    x=Dense(32,activation="relu")(x)
    y=Dense(5,activation="softmax")(x)
    CNN=Model(input,y)
    CNN.compile(optimizer="adadelta",loss="categorical_crossentropy")

    return CNN


def ownModel():
    input_dim=(300,4)
    input1=Input(input_dim)
    input2=Input(input_dim)
    input3 = Input(input_dim)
    input4 = Input(input_dim)
    input5 = Input(input_dim)
    input6 = Input(input_dim)

    con11=Conv1D(4,3,activation='relu')(input1)
    con11=BatchNormalization()(con11)
    con11=Activation("relu")(con11)

    con12=Conv1D(8,5,activation="relu")(con11)
    con12=BatchNormalization()(con12)
    con12=Activation("relu")(con12)

    dense1=Flatten()(con12)
    dense1=Dense(128,activation="relu")(dense1)
#-----------------------------------------------------------

    con21 = Conv1D(4, 3, activation='relu')(input2)
    con21 = BatchNormalization()(con21)
    con21 = Activation("relu")(con21)

    con22 = Conv1D(8, 5, activation="relu")(con21)
    con22 = BatchNormalization()(con22)
    con22 = Activation("relu")(con22)

    dense2 = Flatten()(con22)
    dense2=Dense(128,activation="relu")(dense2)
    # -----------------------------------------------------------

    con31 = Conv1D(4, 3, activation='relu')(input3)
    con31 = BatchNormalization()(con31)
    con31 = Activation("relu")(con31)

    con32 = Conv1D(8, 5, activation="relu")(con31)
    con32 = BatchNormalization()(con32)
    con32 = Activation("relu")(con32)

    dense3 = Flatten()(con32)
    dense3=Dense(128,activation="relu")(dense3)

    # -----------------------------------------------------------


    con41 = Conv1D(4, 3, activation='relu')(input4)
    con41 = BatchNormalization()(con41)
    con41 = Activation("relu")(con41)

    con42 = Conv1D(8, 5, activation="relu")(con41)
    con42 = BatchNormalization()(con42)
    con42 = Activation("relu")(con42)

    dense4 = Flatten()(con42)
    dense4=Dense(128,activation="relu")(dense4)
    # -----------------------------------------------------------


    con51 = Conv1D(4, 3, activation='relu')(input5)
    con51 = BatchNormalization()(con51)
    con51 = Activation("relu")(con51)

    con52 = Conv1D(8, 5, activation="relu")(con51)
    con52 = BatchNormalization()(con52)
    con52 = Activation("relu")(con52)

    dense5 = Flatten()(con52)
    dense5=Dense(128,activation="relu")(dense5)
    # -----------------------------------------------------------


    con61 = Conv1D(4, 3, activation='relu')(input6)
    con61 = BatchNormalization()(con61)
    con61 = Activation("relu")(con61)

    con62 = Conv1D(8, 5, activation="relu")(con61)
    con62 = BatchNormalization()(con62)
    con62 = Activation("relu")(con62)

    dense6 = Flatten()(con62)
    dense6=Dense(128,activation="relu")(dense6)
    # -----------------------------------------------------------

    y=concatenate([dense1,dense2,dense3,dense4,dense5,dense6])
    y=Dense(128,activation="relu")(y)
    y=Dense(32,activation="relu")(y)
    output=Dense(5,activation="softmax")(y)

    ownModel=Model([input1,input2,input3,input4,input5,input6],output)
    ownModel.compile(optimizer="adadelta",loss="categorical_crossentropy")
    return ownModel

def ownModel2():
    input_dim=(300,4)
    input1=Input(input_dim)
    input2=Input(input_dim)
    input3 = Input(input_dim)
    input4 = Input(input_dim)
    input5 = Input(input_dim)
    input6 = Input(input_dim)

    con11=Conv1D(4,3,activation='relu',padding="same")(input1)
    con11=BatchNormalization()(con11)
    con11=Activation("relu")(con11)
    con11=Dropout(0.2)(con11)

    con12=Conv1D(4,5,activation="relu",padding="same")(con11)
    con12=BatchNormalization()(con12)
    con12=Activation("relu")(con12)
    con12=Dropout(0.2)(con12)


    con13=Conv1D(8,7,activation="relu",padding="same")(con12)
    con13=BatchNormalization()(con13)
    con13=Activation("relu")(con13)
    con13=Dropout(0.3)(con13)

    dense1=Flatten()(con13)
    dense1=Dense(128,activation="relu")(dense1)
#-----------------------------------------------------------

    con21 = Conv1D(4, 3, activation='relu',padding="same")(input2)
    con21 = BatchNormalization()(con21)
    con21 = Activation("relu")(con21)

    con22 = Conv1D(4, 5, activation="relu",padding="same")(con21)
    con22 = BatchNormalization()(con22)
    con22 = Activation("relu")(con22)
    con22=Dropout(0.2)(con22)

    con23=Conv1D(8,7,activation="relu",padding="same")(con22)
    con23=BatchNormalization()(con23)
    con23=Activation("relu")(con23)
    con23=Dropout(0.3)(con23)

    dense2 = Flatten()(con23)
    dense2=Dense(128,activation="relu")(dense2)
    # -----------------------------------------------------------

    con31 = Conv1D(4, 3, activation='relu',padding="same")(input3)
    con31 = BatchNormalization()(con31)
    con31 = Activation("relu")(con31)
    con31=Dropout(0.2)(con31)

    con32 = Conv1D(8, 5, activation="relu",padding="same")(con31)
    con32 = BatchNormalization()(con32)
    con32 = Activation("relu")(con32)
    con32=Dropout(0.2)(con32)

    con33=Conv1D(8,7,activation="relu",padding="same")(con32)
    con33=BatchNormalization()(con33)
    con33=Activation("relu")(con33)
    con33=Dropout(0.3)(con33)


    dense3 = Flatten()(con33)
    dense3=Dense(128,activation="sigmoid")(dense3)

    # -----------------------------------------------------------


    con41 = Conv1D(4, 3, activation='relu',padding="same")(input4)
    con41 = BatchNormalization()(con41)
    con41 = Activation("relu")(con41)
    con41=Dropout(0.2)(con41)

    con42 = Conv1D(8, 5, activation="relu",padding="same")(con41)
    con42 = BatchNormalization()(con42)
    con42 = Activation("relu")(con42)
    con42=Dropout(0.2)(con42)

    con43=Conv1D(8,7,activation="relu",padding="same")(con42)
    con43=BatchNormalization()(con43)
    con43=Activation("relu")(con43)
    con43=Dropout(0.3)(con43)


    dense4 = Flatten()(con43)
    dense4=Dense(128,activation="sigmoid")(dense4)
    # -----------------------------------------------------------


    con51 = Conv1D(4, 3, activation='relu',padding="same")(input5)
    con51 = BatchNormalization()(con51)
    con51 = Activation("relu")(con51)
    con51=Dropout(0.2)(con51)

    con52 = Conv1D(4, 5, activation="relu",padding="same")(con51)
    con52 = BatchNormalization()(con52)
    con52 = Activation("relu")(con52)
    con52=Dropout(0.2)(con52)

    con53=Conv1D(8,7,activation="relu",padding="same")(con52)
    con53=BatchNormalization()(con53)
    con53=Activation("relu")(con53)
    con53=Dropout(0.3)(con53)

    dense5 = Flatten()(con53)
    dense5=Dense(128,activation="relu")(dense5)
    # -----------------------------------------------------------


    con61 = Conv1D(4, 3, activation='relu',padding="same")(input6)
    con61 = BatchNormalization()(con61)
    con61 = Activation("relu")(con61)
    con61=Dropout(0.2)(con61)

    con62 = Conv1D(4, 5, activation="relu",padding="same")(con61)
    con62 = BatchNormalization()(con62)
    con62 = Activation("relu")(con62)
    con62=Dropout(0.2)(con62)

    con63=Conv1D(8,7,activation="relu",padding="same")(con62)
    con63=BatchNormalization()(con63)
    con63=Activation("relu")(con63)
    con63=Dropout(0.3)(con63)

    dense6 = Flatten()(con63)
    dense6=Dense(128,activation="relu")(dense6)
    # -----------------------------------------------------------

    y=concatenate([dense1,dense2,dense3,dense4,dense5,dense6])
    y = Dense(128, activation="relu")(y)
    y = Dense(32, activation="relu")(y)
    output=Dense(5,activation="softmax")(y)

    ownModel2=Model([input1,input2,input3,input4,input5,input6],output)
    ownModel2.compile(optimizer="adadelta",loss="categorical_crossentropy")
    return ownModel2