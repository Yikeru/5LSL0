# data_loc = r"C:\Users\20167271\Desktop\ML for signal processing\5LSL0-GIT\Assignment 3\data"
import os
data_loc = os.getcwd()
print(data_loc)
data_loc = data_loc + '\data\Fast_MRI_Knee'
# data_loc = 'D://5LSL0-Datasets//Fast_MRI_Knee' #change the datalocation to something that works for you

batch_size = 64
# print(data_loc)