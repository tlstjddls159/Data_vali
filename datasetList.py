import os
import pandas as pd
import math
import numpy as np
import keras
from keras.layers import Bidirectional, LSTM, CuDNNLSTM, Dropout, Dense, Input, Layer, Conv1D, MaxPooling1D, concatenate, Flatten, CuDNNGRU
from keras.models import Sequential, Model

def dataset(datasetList):

    rpList = []
    for i in os.listdir(datasetList):
        rpList.append(i)
    allDataList = []
    for i in os.listdir(datasetList):
        for j in os.listdir(datasetList + i):
            allDataList.append(datasetList + i + '/' + j)

    allData = []
    for i in allDataList:
        f = open(i, 'r', encoding='UTF-8')
        lines = f.readlines()
        for line in lines:
            allData.append(line)
        f.close()

    bssids = []
    count = 0
    for i in range(len(allData)):
        if "BSSID" in allData[i]:
            print(allData[i].replace("\"BSSID\": ", ''))
            print(allData[i - 1])
            count = count + 1
            if allData[i].replace("\"BSSID\": ", '') not in bssids:
                bssids.append(allData[i].replace("\"BSSID\": ", ''))

    for i in range(len(bssids)):
        bssids[i] = bssids[i].split('\"')[1]

    df = pd.DataFrame(columns = bssids, index = rpList)
    dfTest = pd.DataFrame(columns = bssids, index = rpList)

    for i in df.index:
        for j in os.listdir(datasetList + i):
            f = open(datasetList + i + '/' + j, 'r', encoding='UTF-8')
            lines = f.readlines()
            for k in range(len(lines)):
                if "BSSID" in lines[k] and lines[k].split("\"")[3] in df.columns:
                    if type(df.loc[i][lines[k].split("\"")[3]]) != list:
                        df.loc[i][lines[k].split("\"")[3]] = [float(lines[k + 1].split(':')[1].replace(' ', ''))]
                    else:
                        df.loc[i][lines[k].split("\"")[3]].append(float(lines[k + 1].split(':')[1].replace(' ', '')))


    delcol = df.columns[5:124]

    for i in delcol:
        df = df.drop([i], axis=1)

    # test split
    valList = []
    for i in dfTest.index:
        for j in os.listdir(datasetList + i):
            dfTest = pd.DataFrame(columns = bssids, index = rpList)
            for k in delcol:
                #df = df.drop([i], axis=1)
                dfTest = dfTest.drop([k], axis=1)
            f = open(datasetList + i + '/' + j, 'r', encoding='UTF-8')
            lines = f.readlines()
            for ks in range(len(lines)):
                if "BSSID" in lines[ks] and lines[ks].split("\"")[3] in dfTest.columns:
                    dfTest.loc[i][lines[ks].split("\"")[3]] = float(lines[ks + 1].split(':')[1].replace(' ', ''))

            datass = dfTest.to_numpy(dtype=float)
            datass = np.nan_to_num(datass)
            for alp in range(len(datass)):
                for jet in range(len(datass[alp])):
                    if datass[alp][jet] == 0:
                        datass[alp][jet] = float(-99)

            valList.append(datass)


    for j in df.columns:
        for i in df.index:
            #print(df.loc[i][j])
            if type(df.loc[i][j]) == list:
                df.loc[i][j] = df.loc[i][j]
            else:
                df.loc[i][j] = [-99]

    for j in df.columns:
        for i in df.index:
            sum = 0
            for k in df.loc[i][j]:
                sum = sum + k
            df.loc[i][j] = sum // len(df.loc[i][j])


    x_cal_list = []

    for i in range(156):
        if np.mean(valList[i][0]) != -99:
            x_cal_list.append(valList[i][0])
        elif np.mean(valList[i][1]) != -99:
            x_cal_list.append(valList[i][1])
        elif np.mean(valList[i][2]) != -99:
            x_cal_list.append(valList[i][2])
        elif np.mean(valList[i][3]) != -99:
            x_cal_list.append(valList[i][3])
        elif np.mean(valList[i][4]) != -99:
            x_cal_list.append(valList[i][4])
        elif np.mean(valList[i][5]) != -99:
            x_cal_list.append(valList[i][5])
        elif np.mean(valList[i][6]) != -99:
            x_cal_list.append(valList[i][6])
        elif np.mean(valList[i][7]) != -99:
            x_cal_list.append(valList[i][7])
        elif np.mean(valList[i][8]) != -99:
            x_cal_list.append(valList[i][8])
        elif np.mean(valList[i][9]) != -99:
            x_cal_list.append(valList[i][9])
        elif np.mean(valList[i][10]) != -99:
            x_cal_list.append(valList[i][10])
        elif np.mean(valList[i][11]) != -99:
            x_cal_list.append(valList[i][11])
        elif np.mean(valList[i][12]) != -99:
            x_cal_list.append(valList[i][12])
        elif np.mean(valList[i][13]) != -99:
            x_cal_list.append(valList[i][13])
        elif np.mean(valList[i][14]) != -99:
            x_cal_list.append(valList[i][14])
        elif np.mean(valList[i][15]) != -99:
            x_cal_list.append(valList[i][15])
        elif np.mean(valList[i][16]) != -99:
            x_cal_list.append(valList[i][16])
        elif np.mean(valList[i][17]) != -99:
            x_cal_list.append(valList[i][17])
        elif np.mean(valList[i][18]) != -99:
            x_cal_list.append(valList[i][18])
        elif np.mean(valList[i][19]) != -99:
            x_cal_list.append(valList[i][19])


    y01 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y02 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y03 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y04 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y05 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y06 = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y07 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y08 = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y09 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y10 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y11 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y13 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y14 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    y15 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    y16 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    y17 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    y18 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    y19 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    y20 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


    x_train = df.to_numpy(dtype=float)
    x_train = np.reshape(x_train, (-1, 1,81))
    x_val = np.vstack(x_cal_list)
    x_val = np.reshape(x_val, (-1, 1, 81))


    y_train = np.stack([y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y13, y14, y15, y16, y17, y18, y19, y20])
    y_val = np.stack([y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01, y01,
                      y02, y02, y02, y02, y02, y02, y02, y02, y02, y02, y02, y02, y02, y02, y02, y02, y02, y02,
                      y03, y03, y03, y03, y03, y03,
                      y04, y04, y04, y04, y04, y04,
                      y05, y05, y05, y05, y05, y05,
                      y06, y06, y06, y06, y06, y06,
                      y07, y07, y07, y07, y07, y07,
                      y08, y08, y08, y08, y08, y08,
                      y09, y09, y09, y09, y09, y09,
                      y10, y10, y10, y10, y10, y10,
                      y11, y11, y11, y11, y11, y11,
                      y13, y13, y13, y13, y13, y13, y13, y13, y13, y13, y13, y13,
                      y14, y14, y14, y14, y14, y14,
                      y15, y15, y15, y15, y15, y15,
                      y16, y16, y16, y16, y16, y16,
                      y17, y17, y17, y17, y17, y17,
                      y18, y18, y18, y18, y18, y18,
                      y19, y19, y19, y19, y19, y19,
                      y20, y20, y20, y20, y20, y20])

    return x_train, x_val, y_train, y_val