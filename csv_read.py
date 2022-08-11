import csv
import os
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler


def import_data():
    """csvを読み込む"""
    file = pd.read_csv(os.path.join(os.getcwd(), 'static', 'df_lung_radiomics_ICC_PDL1_fin.csv'))
    print("csv読み込みデータ: {}".format(file.shape))

    # セルに空白が一つでもある行は削除
    file = file.dropna(how='any')
    print("csv読み込みデータ(セルに空白が一つでもある行は削除): {}".format(file.shape))

    # trainデータを作成
    x = file.copy()
    x = x.drop('PDL1', axis=1)
    print("trainデータ: {}".format(x.shape))

    # testデータを作成
    y = file.copy()
    y = y['PDL1']
    print("testデータ: {}".format(y.shape))
    data_list = [x, y]
    return data_list


def corr_spearman_old(df, begin, end):
    """
    各列間の相関係数を算出(Spearman補正)
    :param df: 読み込んだcsvデータ
    :param begin: 補正対象の先頭のカラム名
    :param end: 補正対象の最後のカラム名
    :return: Sprearman補正実施後のdf
    """
    # spearman対象のみに変更
    df = df.loc[:, begin:end]
    print("spearman対象: {}".format(df.shape))
    # 閾値を設定
    threshould = 0.85
    # 相関のある特徴量をリストに抽出
    feat_corr = set()
    corr_matrix = df.corr(method='spearman')
    # 冗長的なカラムをlogへ出力
    # now = datetime.datetime.now()
    # file_name = now.strftime('%Y%m%d%H%M%S') + '_corr_spearman.log'
    # file = os.path.join(os.getcwd(), 'log', file_name)
    # with open(file, mode='w') as f:
    #     writer = csv.writer(f)
    #     for i in range(len(corr_matrix.columns)):
    #         for j in range(i):
    #             if abs(corr_matrix.iloc[i, j]) > threshould:
    #                 feat_name = corr_matrix.columns[i]
    #                 feat_corr.add(feat_name)
    #                 # writer.writerow(feat_name)

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshould:
                feat_name = corr_matrix.columns[i]
                feat_corr.add(feat_name)
                # writer.writerow(feat_name)

    print("冗長性のある特徴量の数: {}".format(len(set(feat_corr))))
    df = df.drop(labels=feat_corr, axis='columns')
    return df


def corr_spearman(df, drop_list=None):
    """
    各列間の相関係数を算出(Spearman補正)
    :param df: 読み込んだcsvデータ
    :param drop_list: Speaman補正対象外カラム
    :return: Spearman補正実施後のdf
    """
    df_org = df.copy()
    # spearman対象のみに変更
    if drop_list is not None:
        # 対象カラムを削除
        df = df.drop(columns=drop_list)
    # 患者ID列は削除する
    df = df.drop(columns=['ID'])
    print("spearman対象: {}".format(df.shape))
    # 閾値を設定
    threshould = 0.85
    # 相関のある特徴量をリストに抽出
    feat_corr = set()
    corr_matrix = df.corr(method='spearman')
    # 冗長的なカラムをlogへ出力
    # now = datetime.datetime.now()
    # file_name = now.strftime('%Y%m%d%H%M%S') + '_corr_spearman.log'
    # file = os.path.join(os.getcwd(), 'log', file_name)
    # with open(file, mode='w') as f:
    #     writer = csv.writer(f)
    #     for i in range(len(corr_matrix.columns)):
    #         for j in range(i):
    #             if abs(corr_matrix.iloc[i, j]) > threshould:
    #                 feat_name = corr_matrix.columns[i]
    #                 feat_corr.add(feat_name)
    #                 # writer.writerow(feat_name)
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > threshould:
                    feat_name = corr_matrix.columns[i]
                    feat_corr.add(feat_name)

    print("冗長性のある特徴量の数: {}".format(len(set(feat_corr))))
    df = df.drop(labels=feat_corr, axis='columns')
    # Speaman補正対象外カラムを加える
    if drop_list is not None:
        for col in drop_list:
            df = pd.concat([df, df_org[col]], axis=1)
    return df


def standardization_zscore(df, drop_list=None):
    """z-scoreで標準化"""
    df_org = df.copy()
    # spearman対象のみに変更
    if drop_list is not None:
        # 対象カラムを削除
        df = df.drop(columns=drop_list)
    df_column = df.columns.tolist()
    stdsc = StandardScaler()
    # Numpy
    df_norm = stdsc.fit_transform(df)
    # Numpy -> DataFrame
    df = pd.DataFrame(data=df_norm, index=df.index, columns=df_column, dtype="float")
    # Speaman補正対象外カラムを加える
    if drop_list is not None:
        for col in drop_list:
            df = pd.concat([df, df_org[col]], axis=1)
    return df
