import numpy as np
import os
'''
if first time run this script, revise t to 1
after that, set t to 2
'''
t=2
if(t==1):
    os.makedirs("./prostate_caner_train_data")
    os.makedirs("./HeLa_train_data")

path="./raw_data"
dataset=["/prostate_cancer","/HeLa"]
prefix=["/circ_","/linear_"]
subtype=["Boundary_","5boundary_","3boundary_","Interior_"]
suffix=["3.fasta","5.fasta"]

DNA_MAP={"A": 1,
         "C": 2,
         "G": 3,
         "T": 4}

OUT_MAP0=[[1,0],
          [0,1]]
OUT_MAP1=[[1,0,0,0,0],  #linear
          [0,1,0,0,0],  #Boundary_
          [0,0,1,0,0],   #5boundary_
          [0,0,0,1,0],   #3boundary_
          [0,0,0,0,1]]      #Interior_

OUT_MAP2=[[1, 0, 0, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 0, 1]]


IN_MAP = np.asarray([
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])




'''
function use to read data from fasta file
'''
def read_fasta_file(file_path):
    f=open(file_path)
    ls=[]
    for line in f:
        if not line.startswith('>'):
            ls.append(line.replace('\n',''))
    f.close()
    return ls
'''
function use to read data from txt file
and transform char to int
'''
def read_txt_file(file_path):
    x_all=[]
    with open(file_path,'r') as f:
        for l in f.readlines():
            line=list(map(int,l.split()))
            x_all.append(line)
    return x_all


'''
transform DNA data to number
'''
def transform_DNA_to_Number(l):
    new_l=list(map(lambda x: DNA_MAP[x], l))
    return new_l
'''
transform X to one-hot vector
'''
def transform_X_to_one_hot(X):
    new_X=list(map(lambda x: IN_MAP[x-1], X))
    return new_X
'''
transform whole x data set to one hot
'''
def transform_x_one_hot_set(x_all):
    x_transform=list(map(transform_X_to_one_hot,x_all))
    return np.asarray(x_transform)
'''
transform whole y data set to one hot
'''
def transform_Y_one_hot_set(Y,type):
    if(type==0):
        new_Y=list(map(lambda x: OUT_MAP0[x[0]], Y))
        return new_Y
    elif(type==1):
        new_Y=list(map(lambda x: OUT_MAP1[x[0]], Y))
        return new_Y
    else:
        new_Y=list(map(lambda x: OUT_MAP2[x[0]], Y))
        return new_Y



'''
function to combine 3' and 5' data together
to form the final data
the first file path is 5' data
the second file path is 3' data
type=1:use to combine x data
type=0: use to combine y
'''
def combine_two_set(file_path_1,file_path_2,type):
    x1=np.asarray(read_txt_file(file_path_1))
    x2=np.asarray(read_txt_file(file_path_2))
    return np.concatenate((x1,x2),axis=type)


'''
Transform raw data to data can be used in Keras
dataset=0: prostate_cancer data
dataset=1: HeLa data
--------------------------------------------------
output_type=0: 2 output type
output_type=1: 5 output type
output_type=2: 8 output type
'''
def create_data_set(output_type,dataset_num):
    def get_y(i,j,output_type):
        if (output_type == 0):
            if i == "/circ_":
                return 1
            else:
                return 0
        elif (output_type == 1):
            if i == "/linear_":
                return 0
            elif i == "/circ_" and j == "Boundary_":
                return 1
            elif i == "/circ_" and j == "5boundary_":
                return 2
            elif i == "/circ_" and j == "3boundary_":
                return 3
            else:
                return 4
        elif (output_type == 2):
            if i == "/linear_" and j=="Boundary_":
                return 0
            elif i=="/linear_" and j=="5boundary_":
                return 1
            elif i=="/linear_" and j=="3boundary_":
                return 2
            elif i=="/linear_" and j=="Interior_":
                return 3
            elif i == "/circ_" and j == "Boundary_":
                return 4
            elif i == "/circ_" and j == "5boundary_":
                return 5
            elif i == "/circ_" and j == "3boundary_":
                return 6
            else:
                return 7
        else:
            raise RuntimeError("output data type should be valid")


    #for loop to read all the file
    datafile=dataset[dataset_num]
    for i in prefix:
        for j in subtype:
            for k in suffix:
                #paste the file path
                file_path=path+datafile+i+j+k
                ls=read_fasta_file(file_path)
                n=len(ls)
                print("The data set: "+file_path+" have %d observation",n)

                '''
                #transform data to form Keras can use
                #some data do not have 600 length, since the precentage
                #is small, simply delete that.
                '''
                x_all = []
                y_all=[]
                for p in range(n):
                    #process x
                    xd=transform_DNA_to_Number(ls[p])
                    if(len(xd)!=600):
                        continue
                    x_all.append(xd)
                    #process y
                    y_all.append(get_y(i,j,output_type))
                x_all=np.asarray(x_all,dtype="int")
                y_all=np.asarray(y_all,dtype="int")
                print(len(x_all))
                print(len(y_all))

                #save data file
                np.savetxt("./"+datafile+"_train_data/"+i+j+k.split(".")[0]+"_X.txt",X=x_all,delimiter=" ",fmt="%i")
                np.savetxt("./"+datafile+"_train_data/"+i+j+k.split(".")[0]+"_Y.txt",X=y_all,delimiter=" ",fmt="%i")

'''
process all the txt data file and form final txt data
used for Keras
'''
def process_train_data(dataset_num):
    x_all=[]
    y_all=[]
    datafile=dataset[dataset_num]
    for i in prefix:
        for j in subtype:
            file_path_1="./"+datafile+"_train_data"+i+j+"5_X.txt"
            file_path_2="./"+datafile+"_train_data"+i+j+"3_X.txt"
            x=combine_two_set(file_path_1,file_path_2,1)
            file_path_3="./"+datafile+"_train_data"+i+j+"5_Y.txt"
            y=np.asarray(read_txt_file(file_path_3))
            if(i=="/circ_"and j=="Boundary_"):
                x_all=x
                y_all=y
            else:
                x_all=np.concatenate((x_all,x),axis=0)
                y_all=np.concatenate((y_all,y),axis=0)
    print(x_all.shape)
    print(y_all)
    np.savetxt("./"+datafile+"_train_data/total_x_8.txt",x_all,delimiter=" ",fmt="%i")
    np.savetxt("./"+datafile + "_train_data/total_y_8.txt", y_all, delimiter=" ", fmt="%i")


