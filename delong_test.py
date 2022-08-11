import datetime
import os

from matplotlib import pyplot as plt
from scipy.stats import norm


def auc(X, Y):
    return 1 / (len(X) * len(Y)) * sum([kernel(x, y) for x in X for y in Y])


def kernel(X, Y):
    return .5 if Y == X else int(Y < X)


def structural_components(X, Y):
    V10 = [1 / len(Y) * sum([kernel(x, y) for y in Y]) for x in X]
    V01 = [1 / len(X) * sum([kernel(x, y) for x in X]) for y in Y]
    return V10, V01


def get_S_entry(V_A, V_B, auc_A, auc_B):
    return 1 / (len(V_A) - 1) * sum([(a - auc_A) * (b - auc_B) for a, b in zip(V_A, V_B)])


def z_score(var_A, var_B, covar_AB, auc_A, auc_B):
    return (auc_A - auc_B) / ((var_A + var_B - 2 * covar_AB) ** .5)


def group_preds_by_label(preds, actual):
    X = [p for (p, a) in zip(preds, actual) if a]
    Y = [p for (p, a) in zip(preds, actual) if not a]
    return X, Y


def delong_main(preds_A, preds_B, preds_C, test_A, test_B, test_C):
    X_A, Y_A = group_preds_by_label(preds_A, test_A)
    X_B, Y_B = group_preds_by_label(preds_B, test_B)
    X_C, Y_C = group_preds_by_label(preds_C, test_C)

    V_A10, V_A01 = structural_components(X_A, Y_A)
    V_B10, V_B01 = structural_components(X_B, Y_B)
    V_C10, V_C01 = structural_components(X_C, Y_C)

    auc_A = auc(X_A, Y_A)
    auc_B = auc(X_B, Y_B)
    auc_C = auc(X_C, Y_C)

    # Compute entries of covariance matrix S (covar_AB = covar_BA)
    var_A = (get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1 / len(V_A10)
             + get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1 / len(V_A01))
    var_B = (get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1 / len(V_B10)
             + get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1 / len(V_B01))
    var_C = (get_S_entry(V_C10, V_C10, auc_C, auc_C) * 1 / len(V_C10)
             + get_S_entry(V_C01, V_C01, auc_C, auc_C) * 1 / len(V_C01))
    covar_AB = (get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1 / len(V_A10)
                + get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1 / len(V_A01))
    covar_BC = (get_S_entry(V_B10, V_C10, auc_B, auc_C) * 1 / len(V_B10)
                + get_S_entry(V_B01, V_C01, auc_B, auc_C) * 1 / len(V_B01))
    covar_CA = (get_S_entry(V_C10, V_A10, auc_C, auc_A) * 1 / len(V_C10)
                + get_S_entry(V_C01, V_A01, auc_C, auc_A) * 1 / len(V_C01))

    # Two tailed test
    z_AB = round(z_score(var_A, var_B, covar_AB, auc_A, auc_B), 3)
    p_AB = round(norm.sf(abs(z_AB)) * 2, 5)
    print()
    print("AB:Delong z-score:", z_AB, "P value:", p_AB)
    print()
    z_BC = round(z_score(var_B, var_C, covar_BC, auc_B, auc_C), 3)
    p_BC = round(norm.sf(abs(z_BC)) * 2, 5)
    print()
    print("BC:Delong z-score:", z_BC, "P value:", p_BC)
    print()
    z_CA = round(z_score(var_C, var_A, covar_CA, auc_C, auc_A), 3)
    p_CA = round(norm.sf(abs(z_CA)) * 2, 5)
    print()
    print("AC:Delong z-score:", z_CA, "P value:", p_CA)
    print()


def auc_forglaph(fpr_A, fpr_B, fpr_C, tpr_A, tpr_B, tpr_C):
    # 2つのAUCを重ねて表示
    fig_max = plt.figure(3, dpi=600)
    # auc_A = metrics.auc(fpr_A, tpr_A)
    # auc_B = metrics.auc(fpr_B, tpr_B)
    # auc_C = metrics.auc(fpr_C, tpr_C)
    # plt.plot(fpr_test, tpr_test, label='ROC curve (area = %.2f)'%auc)
    plt.plot(fpr_A, tpr_A, label="Radiomics", color=[0, 0.4470, 0.7410])
    # plt.plot(fpr_B, tpr_B, label="Clinical")
    # plt.plot(fpr_C, tpr_C, label="Combine")
    plt.legend()
    plt.xlabel("False Positive Rate", fontsize=18, fontname='Times new roman')
    plt.ylabel("True Positive Rate", fontsize=18, fontname='Times new roman')
    plt.grid()
    plt.plot(fpr_B, tpr_B, label="Clinical", color=[0.4660, 0.6740, 0.1880])
    plt.legend()
    # plt.xlabel('FPR: False positive rate')
    # plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.plot(fpr_C, tpr_C, label="Combine", color=[0.8500, 0.3250, 0.0980])
    plt.legend()
    # plt.xlabel('FPR: False positive rate')
    # plt.ylabel('TPR: True positive rate')
    plt.grid()
    # ROC曲線を保存する
    now = datetime.datetime.now()
    file_name = now.strftime('%Y%m%d%H%M%S') + '_Max'
    file = os.path.join(os.getcwd(), 'log', file_name)
    fig_max.savefig(file)
    # ROC曲線を出力
    # plt.show()