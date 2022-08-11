import datetime
import os

import numpy as np
import optuna
import pandas as pd
from boruta import BorutaPy
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold

seed = 999


def featureSelector(x, y):
    """
    特徴量選択
    :param x: 訓練データ
    :param y: 予測対象データ
    :return: 選択された特徴量
    """
    clf = RandomForestClassifier()
    selector = BorutaPy(estimator=clf, n_estimators='auto', two_step=False, verbose=-1, perc=100, max_iter=500,
                        random_state=seed)
    x_array = np.array(x)
    y_array = np.array(y)
    selector.fit(x_array, y_array)
    # 選択された特徴量
    mask = selector.support_
    # print("選択された特徴量：{}".format(mask))

    return mask


def optunaMethod(x_train, y_train, x_val, y_val):
    # Optuna->最小化したいスコアを帰り値とする関数を定義(RandomForest)
    print("--learning Optuna")

    def objective(trial):
        max_depth = trial.suggest_int('max_depth', 1, 25)
        max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 25)
        n_estimators = trial.suggest_int('n_estimators', 2, 100)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 5)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 25)
        rfc = RandomForestClassifier(max_depth=max_depth,
                                     max_leaf_nodes=max_leaf_nodes,
                                     n_estimators=n_estimators,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     random_state=seed)
        rfc.fit(x_train, y_train)
        return roc_auc_score(y_val, rfc.predict(x_val))

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=seed))
    optuna.logging.disable_default_handler()
    study.optimize(objective, n_trials=2000)
    # 最適化したハイパーパラメータの確認
    # print('best_param:{}'.format(study.best_params))
    # 最適化後の目的関数値
    # print('best_value:{}'.format(study.best_value))
    # 最適な試行
    # print('best_trial:{}'.format(study.best_trial))

    return study


# 結果計算
def resultFunction(y, y_pred):
    cnf_matrix = confusion_matrix(y, y_pred)
    accuracy = round(accuracy_score(y, y_pred), 3)
    recall = round(cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0]), 3)
    precision = round(cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[0, 1]), 3)
    specificity = round(cnf_matrix[0, 0] / (cnf_matrix[0, 0] + cnf_matrix[0, 1]), 3)
    auc = round(roc_auc_score(y, y_pred), 3)
    f1 = round(f1_score(y, y_pred), 3)
    result_list = [cnf_matrix, accuracy, recall, precision, specificity, auc, f1]
    return result_list


def featureImportance(x_train, clf):
    """
    重要特徴量を並べる
    :param x_train: 訓練データ
    :param clf: 決定木
    :return:
    """
    # 重要特徴量用Dfを作成
    names, values = x_train.columns, clf.feature_importances_
    impo = []
    for i, feat in enumerate(x_train.columns):
        impo.append(values[i])
        # print('\t{0:10s} : {1:>.6f}'.format(feat, values[i]))
    return impo


def create_roc(fpr, tpr, column_name, cnt, model_type):
    """
    ROC曲線を作成
    :param fpr:
    :param tpr:
    :param column_name:
    :param cnt:
    :param model_type:
    :return:
    """
    plt.rcParams["font.family"] = "Times new roman"
    fig = plt.figure(3, dpi=600)
    plt.plot(fpr, tpr, label=column_name)
    plt.legend(loc='lower right', prop={"family": "Times new roman"})
    if model_type == 'A':
        plt.title("Radomics model", fontsize=18, fontname='Times new roman')
    elif model_type == 'B':
        plt.title("Clinical model", fontsize=18, fontname='Times new roman')
    elif model_type == 'C':
        plt.title("Combined model", fontsize=18, fontname='Times new roman')
    plt.xlabel("False Positive Rate", fontsize=18, fontname='Times new roman')
    plt.ylabel("True Positive Rate", fontsize=18, fontname='Times new roman')
    plt.tick_params(labelsize=13)
    if 5 == cnt:
        # ROC曲線を保存する
        now = datetime.datetime.now()
        file_name = now.strftime('%Y%m%d%H%M%S') + '_' + model_type + '_rocCurve.png'.format(cnt)
        file = os.path.join(os.getcwd(), 'log', file_name)
        fig.savefig(file)
        # ROC曲線を出力
        plt.show()


def kFoldCrossVal(x, y, model_type, drop_list=None):
    """
    K-Fold交差検証
    :param x: 訓練データ
    :param y: 予測対象データ
    :param model_type: model種類
    :param drop_list: 削除対象カラム
    :return:
    """
    # csv出力リスト
    performance_index = ["train_num", "test_num", "train", "test", "model", "test_pred"]
    r_index = ["y_train_cnt", "y_validation_cnt", "y_train1_cnt",
               "y_validation1_cnt", "model", "cnf_matrix_train", "accuracy_train", "Recall_train",
               "Precision_train", "Specificity_train", "AUC_train", "F1_score_train", "cnf_matrix_vali",
               "accuracy_vali", "Recall_vali", "Precision_vali", "Specificity_vali", "AUC_vali",
               "F1_score_vali"]
    # 結果格納用DataFrame
    p_df = pd.DataFrame(index=r_index)
    i_df = pd.DataFrame()
    roc_fpr_df = pd.DataFrame()
    roc_tpr_df = pd.DataFrame()
    proba_df = pd.DataFrame()

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cnt = 1  # loop回数カウント
    auc_ave = 0
    for row_train, row_validation in kf.split(x, y):
        print("-----------------{}回目-------------------".format(cnt))
        # trainデータ：validationデータ = (k-1):k
        x_train, x_validation = x.iloc[row_train, :], x.iloc[row_validation, :]
        x_train_org = x_train.copy()
        x_validation_org = x_validation.copy()
        y_train, y_validation = y[row_train], y[row_validation]
        if drop_list is not None:
            # 対象カラムを削除
            x_train = x_train.drop(columns=drop_list)
            x_validation = x_validation.drop(columns=drop_list)
        # 結果リストに格納
        result_list = [len(y_train), len(y_validation), (y_train == 1).values.sum(), (y_validation == 1).values.sum()]

        # Borutaの特徴量選択
        mask = featureSelector(x_train, y_train)
        # 選択された特徴量のデータセットを作成
        x_train = x_train.iloc[:, mask]
        x_validation = x_validation.iloc[:, mask]
        if drop_list is not None:
            # 対象カラムを追加
            for col in drop_list:
                x_train = pd.concat([x_train, x_train_org[col]], axis=1)
                x_validation = pd.concat([x_validation, x_validation_org[col]], axis=1)
        # optunaでパラメータチューニング
        study = optunaMethod(x_train, y_train, x_validation, y_validation)
        # 結果をlogへ出力
        now = datetime.datetime.now()
        # x_trainを出力
        tra_file_name = now.strftime('%Y%m%d%H%M%S') + '_train{}{}.csv'.format(cnt, model_type)
        tra_file = os.path.join(os.getcwd(), 'log', 'data', tra_file_name)
        x_train.to_csv(tra_file)
        # x_validationを出力
        val_file_name = now.strftime('%Y%m%d%H%M%S') + '_validation{}{}.csv'.format(cnt, model_type)
        val_file = os.path.join(os.getcwd(), 'log', 'data', val_file_name)
        x_validation.to_csv(val_file)
        # 最適化したパラメータでチューニング
        clf = RandomForestClassifier(max_depth=study.best_params['max_depth'],
                                     max_leaf_nodes=study.best_params['max_leaf_nodes'],
                                     n_estimators=study.best_params['n_estimators'],
                                     min_samples_split=study.best_params['min_samples_split'],
                                     min_samples_leaf=study.best_params['min_samples_leaf'],
                                     random_state=seed)
        # モデル作成
        model = clf.fit(x_train, y_train)
        # trainデータで結果を予測
        y_train_pred = model.predict(x_train)
        y_validation_pred = model.predict(x_validation)
        # 結果リストに格納
        result_list.append(model)
        result_list.extend(resultFunction(y_train, y_train_pred))
        result_list.extend(resultFunction(y_validation, y_validation_pred))

        # 列名を作成
        column_name = "model_{}".format(cnt)
        # 精度出力用Dfを作成
        performance_series = pd.Series(result_list, index=r_index, name=column_name)
        p_df = pd.concat([p_df, performance_series], axis=1)
        # 重要特徴量用Dfを作成
        impo = featureImportance(x_train, clf)
        importance_series = pd.Series(impo, index=x_train.columns, name=column_name)
        # importance_series = importance_series.sort_values(ascending=False)
        column_name_df = [column_name]
        i_df = pd.concat(
            [i_df, pd.DataFrame(impo, index=x_train.columns, columns=column_name_df)], axis=1)
        # ROC曲線用Dfを作成
        result_auc = model.predict_proba(x_validation)[:, 1]
        # proba_df = pd.concat([proba_df, pd.DataFrame(result_auc, columns=column_name_df)], axis=1)
        fpr, tpr, thresholds = roc_curve(y_validation, model.predict_proba(x_validation)[:, 1],
                                         drop_intermediate=False)
        # auc_value = auc(fpr, tpr)
        create_roc(fpr, tpr, column_name, cnt, model_type)

        # K回のクロスバリデーションの中で一番良いモデルを保存する
        k_auc = round(roc_auc_score(y_validation, y_validation_pred), 3)
        if 1 == cnt:  # 1回目はとりあえず格納する
            best_auc = k_auc
            best_cnt = cnt
            max_importance_series = importance_series
            max_preds = result_auc
            max_val = y_validation
            max_fpr = fpr
            max_tpr = tpr
        elif best_auc < k_auc:  # 前のAUCの値より大きい場合
            best_auc = k_auc
            best_cnt = cnt
            max_importance_series = importance_series
            max_preds = result_auc
            max_val = y_validation
            max_fpr = fpr
            max_tpr = tpr
        cnt += 1
        auc_ave += k_auc
    print("best AUC: {} (model{})".format(best_auc, best_cnt))
    delong_list = [max_preds, max_val, max_fpr, max_tpr]
    # 結果をlogへ出力
    now = datetime.datetime.now()
    # 精度
    p_file_name = now.strftime('%Y%m%d%H%M%S') + '_performance' + model_type + '.csv'
    p_file = os.path.join(os.getcwd(), 'log', p_file_name)
    p_df.to_csv(p_file)
    # 重要特徴量
    i_file_name = now.strftime('%Y%m%d%H%M%S') + '_importance' + model_type + '.csv'
    i_file = os.path.join(os.getcwd(), 'log', i_file_name)
    i_df.to_csv(i_file)

    return auc_ave/5, delong_list


def kFoldCrossVal_cli(x, y, model_type, start_column=None, end_column=None):
    """
    K-Fold交差検証
    :param x: 訓練データ
    :param y: 予測対象データ
    :param model_type: model種類
    :param start_column:
    :param end_column:
    :return:
    """
    # csv出力リスト
    performance_index = ["train_num", "test_num", "train", "test", "model", "test_pred"]
    r_index = ["y_train_cnt", "y_validation_cnt", "y_train1_cnt",
               "y_validation1_cnt", "model", "cnf_matrix_train", "accuracy_train", "Recall_train",
               "Precision_train", "Specificity_train", "AUC_train", "F1_score_train", "cnf_matrix_vali",
               "accuracy_vali", "Recall_vali", "Precision_vali", "Specificity_vali", "AUC_vali",
               "F1_score_vali"]
    # 結果格納用DataFrame
    p_df = pd.DataFrame(index=r_index)
    i_df = pd.DataFrame()
    roc_fpr_df = pd.DataFrame()
    roc_tpr_df = pd.DataFrame()
    proba_df = pd.DataFrame()

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cnt = 1  # loop回数カウント
    auc_ave = 0
    for row_train, row_validation in kf.split(x, y):
        print("-----------------{}回目-------------------".format(cnt))
        # trainデータ：validationデータ = (k-1):k
        x_train, x_validation = x.iloc[row_train, :], x.iloc[row_validation, :]
        x_train_org = x_train.copy()
        x_validation_org = x_validation.copy()
        y_train, y_validation = y[row_train], y[row_validation]
        # if drop_list is not None:
        #     # 対象カラムを削除
        #     x_train = x_train.drop(columns=drop_list)
        #     x_validation = x_validation.drop(columns=drop_list)
        if start_column is not None and end_column is not None:
            # 対象カラムのみに絞り込む
            x_train = x_train.loc[:, start_column:end_column]
            x_validation = x_validation.loc[:, start_column:end_column]
        # 結果リストに格納
        result_list = [len(y_train), len(y_validation), (y_train == 1).values.sum(), (y_validation == 1).values.sum()]

        # # Borutaの特徴量選択
        # mask = featureSelector(x_train, y_train)
        # # 選択された特徴量のデータセットを作成
        # x_train = x_train.iloc[:, mask]
        # x_validation = x_validation.iloc[:, mask]
        # if add_list is not None:
        #     # 対象カラムを追加
        #     for col in add_list:
        #         x_train = pd.concat([x_train, x_train_org[col]], axis=1)
        #         x_validation = pd.concat([x_validation, x_validation_org[col]], axis=1)
        # optunaでパラメータチューニング
        study = optunaMethod(x_train, y_train, x_validation, y_validation)
        # 結果をlogへ出力
        now = datetime.datetime.now()
        # x_trainを出力
        tra_file_name = now.strftime('%Y%m%d%H%M%S') + '_train{}{}.csv'.format(cnt, model_type)
        tra_file = os.path.join(os.getcwd(), 'log', 'data', tra_file_name)
        x_train.to_csv(tra_file)
        # x_validationを出力
        val_file_name = now.strftime('%Y%m%d%H%M%S') + '_validation{}{}.csv'.format(cnt, model_type)
        val_file = os.path.join(os.getcwd(), 'log', 'data', val_file_name)
        x_validation.to_csv(val_file)
        # 最適化したパラメータでチューニング
        clf = RandomForestClassifier(max_depth=study.best_params['max_depth'],
                                     max_leaf_nodes=study.best_params['max_leaf_nodes'],
                                     n_estimators=study.best_params['n_estimators'],
                                     min_samples_split=study.best_params['min_samples_split'],
                                     min_samples_leaf=study.best_params['min_samples_leaf'],
                                     random_state=seed)
        # モデル作成
        model = clf.fit(x_train, y_train)
        # trainデータで結果を予測
        y_train_pred = model.predict(x_train)
        y_validation_pred = model.predict(x_validation)
        # 結果リストに格納
        result_list.append(model)
        result_list.extend(resultFunction(y_train, y_train_pred))
        result_list.extend(resultFunction(y_validation, y_validation_pred))

        # 列名を作成
        column_name = "model_{}".format(cnt)
        # 精度出力用Dfを作成
        performance_series = pd.Series(result_list, index=r_index, name=column_name)
        p_df = pd.concat([p_df, performance_series], axis=1)
        # 重要特徴量用Dfを作成
        impo = featureImportance(x_train, clf)
        importance_series = pd.Series(impo, index=x_train.columns, name=column_name)
        # importance_series = importance_series.sort_values(ascending=False)
        column_name_df = [column_name]
        i_df = pd.concat(
            [i_df, pd.DataFrame(impo, index=x_train.columns, columns=column_name_df)], axis=1)
        # ROC曲線用Dfを作成
        result_auc = model.predict_proba(x_validation)[:, 1]
        # proba_df = pd.concat([proba_df, pd.DataFrame(result_auc, columns=column_name_df)], axis=1)
        fpr, tpr, thresholds = roc_curve(y_validation, model.predict_proba(x_validation)[:, 1],
                                         drop_intermediate=False)
        # auc_value = auc(fpr, tpr)
        create_roc(fpr, tpr, column_name, cnt, model_type)

        # K回のクロスバリデーションの中で一番良いモデルを保存する
        k_auc = round(roc_auc_score(y_validation, y_validation_pred), 3)
        if 1 == cnt:  # 1回目はとりあえず格納する
            best_auc = k_auc
            best_cnt = cnt
            max_importance_series = importance_series
            max_preds = result_auc
            max_val = y_validation
            max_fpr = fpr
            max_tpr = tpr
        elif best_auc < k_auc:  # 前のAUCの値より大きい場合
            best_auc = k_auc
            best_cnt = cnt
            max_importance_series = importance_series
            max_preds = result_auc
            max_val = y_validation
            max_fpr = fpr
            max_tpr = tpr
        cnt += 1
        auc_ave += k_auc
    print("best AUC: {} (model{})".format(best_auc, best_cnt))
    delong_list = [max_preds, max_val, max_fpr, max_tpr]
    # 結果をlogへ出力
    now = datetime.datetime.now()
    # 精度
    p_file_name = now.strftime('%Y%m%d%H%M%S') + '_performance' + model_type + '.csv'
    p_file = os.path.join(os.getcwd(), 'log', p_file_name)
    p_df.to_csv(p_file)
    # 重要特徴量
    i_file_name = now.strftime('%Y%m%d%H%M%S') + '_importance' + model_type + '.csv'
    i_file = os.path.join(os.getcwd(), 'log', i_file_name)
    i_df.to_csv(i_file)

    return auc_ave/5, delong_list
