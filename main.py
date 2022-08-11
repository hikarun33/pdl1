import datetime
import os
import pandas as pd

import csv_read as rd
import learning as ln
import delong_test as dg


def main():
    os.mkdir(os.path.join(os.getcwd(), 'log', 'data'))

    # Radiomicsモデル
    print("Radiomics Model")
    # データ読み込み
    read_file = rd.import_data()
    delList_A = ['Sex', 'Solid size', 'Smoking']
    # trainデータ
    x_read_file = read_file[0].drop(columns=delList_A)
    # testデータ
    y_read_file = read_file[1].drop(columns=delList_A)
    # Spearman補正
    x_read_file = rd.corr_spearman(x_read_file)
    # z-scoreで標準化
    x_read_file = rd.standardization_zscore(x_read_file)
    # K-fold交差検証
    result = ln.kFoldCrossVal(x_read_file, y_read_file, 'A')
    best_cntA = result[0]
    delong_listA = result[1]

    # clinicalモデル
    print("Clinical Model")
    # csvデータ
    read_file = rd.import_data()
    # trainデータ
    x_read_file = read_file[0]
    # testデータ
    y_read_file = read_file[1]
    # K-fold交差検証
    result = ln.kFoldCrossVal_cli(x_read_file, y_read_file, 'B', 'Sex', 'Smoking')
    best_cntB = result[0]
    delong_listB = result[1]

    # combinedモデル
    print("Combined Model")
    # csvデータ
    read_file = rd.import_data()
    # trainデータ
    x_read_file = read_file[0]
    # testデータ
    y_read_file = read_file[1]
    # Spearman補正
    delList_C = ['Sex', 'Solid size', 'Smoking']
    x_read_file = rd.corr_spearman(x_read_file, delList_C)
    # z-scoreで標準化
    x_read_file = rd.standardization_zscore(x_read_file, delList_C)
    # K-fold交差検証
    result = ln.kFoldCrossVal(x_read_file, y_read_file, 'C', delList_C)
    best_cntC = result[0]
    delong_listC = result[1]

    print("Average AUC Score")
    print("Model_A: {}".format(best_cntA))
    print("Model_B: {}".format(best_cntB))
    print("Model_C: {}".format(best_cntC))

    # delongテスト
    dg.delong_main(delong_listA[0], delong_listB[0], delong_listC[0], delong_listA[1], delong_listB[1], delong_listC[1])
    dg.auc_forglaph(delong_listA[2], delong_listB[2], delong_listC[2], delong_listA[3], delong_listB[3], delong_listC[3])


if __name__ == '__main__':
    main()
