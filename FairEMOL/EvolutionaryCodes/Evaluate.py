import numpy as np
from itertools import product
import copy
from sklearn.metrics import log_loss
from scipy.spatial.distance import pdist


def get_label(logits):
    pred_label = logits
    pred_label[np.where(pred_label >= 0.5)] = 1
    pred_label[np.where(pred_label < 0.5)] = 0
    pred_label = pred_label.reshape(1, logits.shape[0] * logits.shape[1])
    pred_label = pred_label.reshape(1, -1)
    return pred_label


def get_average(group_values, plan):
    if plan == 1:
        values = []
        num_group = len(group_values)
        if num_group > 1:
            for i in range(num_group):
                if i == (num_group - 1):
                    break
                for j in range(i + 1, num_group):
                    values.append(np.abs(group_values[i] - group_values[j]))

            return np.mean(values)
        else:
            return 0
    else:
        values = 0.0
        num_group = len(group_values)
        if num_group > 1:
            for i in range(num_group):
                if i == (num_group - 1):
                    break
                for j in range(i + 1, num_group):
                    values = np.max([values, np.abs(group_values[i] - group_values[j])])

            return values
        else:
            return 0


def get_obj(group_values, plan):
    Group_values = copy.deepcopy(group_values)
    if plan == 1:
        # calculate the difference
        values = []
        num_group = len(group_values)
        if num_group > 1:
            for i in range(num_group):
                if i == (num_group - 1):
                    break
                for j in range(i + 1, num_group):
                    values.append(np.abs(group_values[i] - group_values[j]))

            return 0.5 * (np.mean(values) + np.max(values))

        elif num_group == 1:
            return 1

        else:
            return 0
    else:
        # calculate the ratio
        values = []
        num_group = len(group_values)
        if num_group > 1:
            for i in range(num_group):
                if i == (num_group - 1):
                    break
                for j in range(i + 1, num_group):
                    if group_values[j] == 0 and group_values[i] == 0:
                        values.append(1)
                    elif group_values[j] == 0 and group_values[i] != 0:
                        values.append(0)
                    elif group_values[j] != 0 and group_values[i] == 0:
                        values.append(0)
                    else:
                        values.append(
                            np.min([(group_values[j] / group_values[i]), (group_values[i] / group_values[j])]))
            return 0.5 * (1 - np.mean(values) + 1 - np.min(values))

        elif num_group == 1:
            return 1

        else:
            return 0


def calcul_all_fairness_objs(data, data_norm, logits, truelabel, sensitive_attributions, alpha, obj_names,
                             privileged_class_names, isprivileged=1):
    logits = logits.reshape(1, -1).astype(np.float64)
    truelabel = truelabel.astype(np.float64).reshape(1, -1)
    pred_label = get_label(logits.copy())
    privileged = []
    for s in sensitive_attributions:
        privileged.append(privileged_class_names[s])

    benefits = logits - truelabel + 1  # new version in section 3.1
    benefits_group = copy.deepcopy(benefits)

    total_num = logits.shape[1]

    attribution = data.columns
    group_attr = []
    check_gmuns = []
    group_dict = {}

    Disparate_impact = []
    Calibration_Neg = []
    Predictive_parity = []
    Discovery_ratio = []
    Discovery_diff = []
    Predictive_equality = []
    FPR_ratio = []
    Equal_opportunity = []
    Equalized_odds1 = []
    Equalized_odds2 = []
    Average_odd_diff = []
    Conditional_use_accuracy_equality1 = []
    Conditional_use_accuracy_equality2 = []
    Overall_accuracy = []
    Error_ratio = []
    Error_diff = []
    Statistical_parity = []
    FOR_ratio = []
    FOR_diff = []
    FPR_ratio = []
    FNR_ratio = []
    FNR_diff = []

    favorable_class = ''
    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
                if '_' + privileged[0] in attr:
                    favorable_class = attr
        group_dict.update({sens: temp})

    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])
    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, truelabel.shape[1]]) == np.ones([1, truelabel.shape[1]])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            check_gmuns.append(g_num)
            g_idx = np.array(np.where(flag)).reshape([1, -1])[0]
            benefits_group[0, g_idx] = np.mean(benefits[0, g_idx])
            # g_logits = logits[0, g_idx]
            g_truelabel = truelabel[0, g_idx]
            g_predlabel = pred_label[0, g_idx]

            # P(d=1 | g)
            # Disparate Impact  or  Statistical Parity
            if "Disparate_impact" in obj_names or "Statistical_parity" in obj_names:
                Disparate_impact.append(np.sum(g_predlabel) / g_num)
                Statistical_parity = Disparate_impact

            # P(y=d | g)
            # Overall accuracy
            if "Overall_accuracy" in obj_names:
                Overall_accuracy.append(np.sum(g_truelabel == g_predlabel) / g_num)

            # P(y != d, g)
            # Error ratio   or   Error diff
            if "Error_ratio" in obj_names or "Error_diff" in obj_names:
                Error_ratio.append(1 - np.sum(g_truelabel == g_predlabel) / total_num)
                Error_diff = Error_ratio

            # P(y=1 | d=1, g)
            # Predictive parity
            if "Predictive_parity" in obj_names:
                if np.sum(g_predlabel) > 0:
                    Predictive_parity.append(np.sum(g_truelabel * g_predlabel) / np.sum(g_predlabel))

            # P(y=0 | d=1, g)
            # Discovery ratio  or   Discovery diff
            if "Discovery_ratio" in obj_names or "Discovery_diff" in obj_names:
                if np.sum(g_predlabel) > 0:
                    Discovery_ratio.append((np.sum((1 - g_truelabel) * g_predlabel) / np.sum(g_predlabel)))
                    Discovery_diff = Discovery_ratio

            # P(y=1 | d=0, g)
            # Calibration-   or   FOR ratio    or   FOR diff
            if "Calibration_neg" in obj_names or "FOR_ratio" in obj_names or "FOR_diff" in obj_names:
                if np.sum(1 - g_predlabel) > 0:
                    Calibration_Neg.append(np.sum(g_truelabel * (1 - g_predlabel)) / np.sum(1 - g_predlabel))
                    FOR_ratio = Calibration_Neg
                    FOR_diff = Calibration_Neg

            # P(d=1 | y=0, g)
            # Predictive equality   or   FPR ratio
            if "Predictive_equality" in obj_names or "FPR_ratio" in obj_names:
                if np.sum(1 - g_truelabel) > 0:
                    Predictive_equality.append(np.sum(g_predlabel * (1 - g_truelabel)) / np.sum(1 - g_truelabel))
                    FPR_ratio = Predictive_equality

            # P(d=1 | y=1, g)
            # Equal opportunity
            if "Equal_opportunity" in obj_names:
                if np.sum(g_truelabel) > 0:
                    Equal_opportunity.append(np.sum(g_predlabel * g_truelabel) / np.sum(g_truelabel))

            # P(d=0 | y=1, g)
            # FNR ratio    or    FNR diff
            if "FNR_ratio" in obj_names or "FNR_diff" in obj_names:
                if np.sum(g_truelabel) > 0:
                    FNR_ratio.append(np.sum((1 - g_predlabel) * g_truelabel) / np.sum(g_truelabel))
                    FNR_diff = FNR_ratio

            # P(d=1 | y=0, g) and P(d=1 | y=1, g)
            # Equalized odds
            if "Equalized_odds" in obj_names:
                if np.sum(g_truelabel) > 0:
                    Equalized_odds1.append(np.sum(g_predlabel * g_truelabel) / np.sum(g_truelabel))
                if np.sum(1 - g_truelabel) > 0:
                    Equalized_odds2.append(np.sum(g_predlabel * (1 - g_truelabel)) / np.sum(1 - g_truelabel))

            # Conditional use accuracy equality
            if "Conditional_use_accuracy_equality" in obj_names:
                if np.sum(g_predlabel) > 0:
                    Conditional_use_accuracy_equality1.append(np.sum(g_truelabel * g_predlabel) / np.sum(g_predlabel))
                if np.sum(1 - g_predlabel) > 0:
                    Conditional_use_accuracy_equality2.append(
                        np.sum((1 - g_truelabel) * (1 - g_predlabel)) / np.sum(1 - g_predlabel))

            # P(d=1 | y=0, g) + P(d=1 | y=1, g)
            # Average odd difference
            if "Average_odd_diff" in obj_names:
                if np.sum(g_truelabel) > 0 and np.sum(1 - g_truelabel) > 0:
                    Average_odd_diff.append((np.sum(g_predlabel * g_truelabel) / np.sum(g_truelabel)) + (
                            np.sum(g_predlabel * (1 - g_truelabel)) / np.sum(1 - g_truelabel)))

    Groups_info = {}
    if "Accuracy" in obj_names:
        Groups_info.update({"Accuracy": np.mean(pred_label == truelabel)})

    # BCE loss
    if "Error" in obj_names:
        BCE_loss = log_loss(truelabel.reshape(-1, 1), np.hstack([1 - logits.reshape(-1, 1), logits.reshape(-1, 1)]))
        Groups_info.update({"Error": BCE_loss})

    # Individual unfairness = within-group + between-group
    if "Individual_fairness" in obj_names:
        Individual_fairness_val = generalized_entropy_index(benefits, alpha)
        Groups_info.update({"Individual_fairness": Individual_fairness_val})

    # Group unfairness = between-group
    if "Group_fairness" in obj_names:
        Group_fairness_val = generalized_entropy_index(benefits_group, alpha)
        Groups_info.update({"Group_fairness": Group_fairness_val})

    # Disparate impact
    if "Disparate_impact" in obj_names:
        Disparate_impact_val = get_obj(Disparate_impact, 2)
        Groups_info.update({"Disparate_impact": Disparate_impact_val})

    # Statistical parity
    if "Statistical_parity" in obj_names:
        Statistical_parity_val = get_obj(Statistical_parity, 1)
        Groups_info.update({"Statistical_parity": Statistical_parity_val})

    # Overall accuracy
    if "Overall_accuracy" in obj_names:
        Overall_accuracy_val = get_obj(Overall_accuracy, 1)
        Groups_info.update({"Overall_accuracy": Overall_accuracy_val})

    # Error ratio
    if "Error_ratio" in obj_names:
        Error_ratio_val = get_obj(Error_ratio, 2)
        Groups_info.update({"Error_ratio": Error_ratio_val})

    # Error diff
    if "Error_diff" in obj_names:
        Error_diff_val = get_obj(Error_diff, 1)
        Groups_info.update({"Error_diff": Error_diff_val})

    # Predictive parity
    if "Predictive_parity" in obj_names:
        Predictive_parity_val = get_obj(Predictive_parity, 1)
        Groups_info.update({"Predictive_parity": Predictive_parity_val})

    # Discovery ratio
    if "Discovery_ratio" in obj_names:
        Discovery_ratio_val = get_obj(Discovery_ratio, 2)
        Groups_info.update({"Discovery_ratio": Discovery_ratio_val})

    # Discovery diff
    if "Discovery_diff" in obj_names:
        Discovery_diff_val = get_obj(Discovery_diff, 1)
        Groups_info.update({"Discovery_diff": Discovery_diff_val})

    # Calibration Neg
    if "Calibration_neg" in obj_names:
        Calibration_Neg_val = get_obj(Calibration_Neg, 1)
        Groups_info.update({"Calibration_neg": Calibration_Neg_val})

    # FOR ratio
    if "FOR_ratio" in obj_names:
        FOR_ratio_val = get_obj(FOR_ratio, 2)
        Groups_info.update({"FOR_ratio": FOR_ratio_val})

    # FOR diff
    if "FOR_diff" in obj_names:
        FOR_diff_val = get_obj(FOR_diff, 1)
        Groups_info.update({"FOR_diff": FOR_diff_val})

    # Predictive equality
    if "Predictive_equality" in obj_names:
        Predictive_equality_val = get_obj(Predictive_equality, 1)
        Groups_info.update({"Predictive_equality": Predictive_equality_val})

    # FPR ratio
    if "FPR_ratio" in obj_names:
        FPR_ratio_val = get_obj(FPR_ratio, 2)
        Groups_info.update({"FPR_ratio": FPR_ratio_val})

    # Equal opportunity
    if "Equal_opportunity" in obj_names:
        Equal_opportunity_val = get_obj(Equal_opportunity, 1)
        Groups_info.update({"Equal_opportunity": Equal_opportunity_val})

    # FNR ratio
    if "FNR_ratio" in obj_names:
        FNR_ratio_val = get_obj(FNR_ratio, 2)
        Groups_info.update({"FNR_ratio": FNR_ratio_val})

    # FNR diff
    if "FNR_diff" in obj_names:
        FNR_diff_val = get_obj(FNR_diff, 1)
        Groups_info.update({"FNR_diff": FNR_diff_val})

    # Equalized odds
    if "Equalized_odds" in obj_names:
        Equalized_odds_val = 0.5 * (get_obj(Equalized_odds1, 1) + get_obj(Equalized_odds2, 1))
        Groups_info.update({"Equalized_odds": Equalized_odds_val})

    # Conditional use accuracy equality
    if "Conditional_use_accuracy_equality" in obj_names:
        Conditional_use_accuracy_equality_val = 0.5 * (
                get_obj(Conditional_use_accuracy_equality1, 1) + get_obj(Conditional_use_accuracy_equality2, 1))
        Groups_info.update({"Conditional_use_accuracy_equality": Conditional_use_accuracy_equality_val})

    # Average odd difference
    if "Average_odd_diff" in obj_names:
        Average_odd_diff_val = 0.5 * get_obj(Average_odd_diff, 1)
        Groups_info.update({"Average_odd_diff": Average_odd_diff_val})

    return Groups_info


def calcul_simple_metric(info, base_name, type="diff"):
    if type == "diff":
        g1 = info[0][base_name]   # priviledged
        g2 = info[1][base_name]   # unpriviledged
        return g2 - g1
    else:
        g1 = info[0][base_name]
        g2 = info[1][base_name]
        if g1 == 0 and g2 == 0:
            return 1
        if g1 == 0:
            return 0
        return g2 / g1



def calcul_all_fairness_objs_new(data, data_norm, logits, truelabel, sensitive_attributions, alpha, obj_names,
                                 privileged_class_names, data_dist=None):
    # 这是一个基于calcul_all_fairness_objs函数修改的一个版本，具体更新如下：
    # 1. 仅考虑两类：privileged、unprivileged
    # 2. 为保证计算结果的一致性，与AIF360保持一致
    # 3. 考虑更多的公平性指标
    # 对于待预测的标签： Favorable类别对应truelabel中的1

    logits = logits.reshape(1, -1).astype(np.float64)
    truelabel = truelabel.astype(np.float64).reshape(1, -1)
    pred_label = get_label(logits.copy())

    benefits = logits - truelabel + 1  # new version in section 3.1
    benefits_group = copy.deepcopy(benefits)
    benefits_group[0, :] = 0

    attribution = data.columns
    group_attr = []
    group_dict = {}
    group_values = []

    privileged = []
    for s in sensitive_attributions:
        privileged.append(privileged_class_names[s])
    favorable_class = ''
    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
                if '_' + str(privileged[0]) in attr:
                    favorable_class = attr
        group_dict.update({sens: temp})

    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])

    # ######## group_attr[0][0]: privileged, group_attr[0][1]: unprivileged
    # ######## favorable label: 1, unfavorable: 0
    flag = np.ones([1, truelabel.shape[1]]) == np.ones([1, truelabel.shape[1]])
    privileged_flag = flag & data[favorable_class]
    unprivileged_flag = ~privileged_flag
    flags = []
    flags.append(privileged_flag)
    flags.append(unprivileged_flag)

    for flag in flags:
        g_num = np.sum(flag)

        if g_num != 0:
            # check_gmuns.append(g_num)
            g_idx = np.array(np.where(flag)).reshape([1, -1])[0]
            g_logits = logits[0, g_idx]
            g_truelabel = truelabel[0, g_idx]
            g_predlabel = pred_label[0, g_idx]
            benefits_group[0, g_idx] = np.mean(g_predlabel - g_truelabel + 1)

            TP = np.sum(g_truelabel * g_predlabel, dtype=np.float64)
            FP = np.sum((1.0 - g_truelabel) * g_predlabel, dtype=np.float64)
            TN = np.sum((1.0 - g_truelabel) * (1.0 - g_predlabel), dtype=np.float64)
            FN = np.sum(g_truelabel * (1.0 - g_predlabel), dtype=np.float64)
            GTP = np.sum(g_truelabel * g_logits, dtype=np.float64)
            GFP = np.sum((1.0 - g_truelabel) * g_logits, dtype=np.float64)
            GTN = np.sum((1.0 - g_truelabel) * (1.0 - g_logits), dtype=np.float64)
            GFN = np.sum(g_truelabel * (1.0 - g_logits), dtype=np.float64)
            bene_log = np.mean(benefits[0, g_idx])
            P = np.sum(g_truelabel, dtype=np.float64)
            N = np.sum(1 - g_truelabel, dtype=np.float64)
            ACC = (TP + TN) / (P + N) if (P + N) > 0.0 else np.float64(0.0)

            temp_dict = dict(
                TP=TP, FP=FP, TN=TN, FN=FN,
                GTP=GTP, GFP=GFP, GTN=GTN, GFN=GFN,
                bene_log=bene_log, pred1=np.sum(g_predlabel),
                P=P, N=N,
                TPR=TP / P, TNR=TN / N, FPR=FP / N, FNR=FN / P,
                GTPR=GTP / P, GTNR=GTN / N, GFPR=GFP / N, GFNR=GFN / P,
                PPV=TP / (TP + FP) if (TP + FP) > 0.0 else np.float64(0.0),
                NPV=TN / (TN + FN) if (TN + FN) > 0.0 else np.float64(0.0),
                FDR=FP / (FP + TP) if (FP + TP) > 0.0 else np.float64(0.0),
                FOR=FN / (FN + TN) if (FN + TN) > 0.0 else np.float64(0.0),
                Selection_rate=np.sum(g_predlabel) / g_num,
                ACC=ACC,
                ERR=1.0 - ACC,
                totla_num=g_num
            )

            group_values.append(temp_dict)
    # ****** 0. true_positive_rate_difference ******  optimal = 0
    r""":math:`TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}`
    """
    true_positive_rate_difference = calcul_simple_metric(group_values, "TPR", "diff")
    true_positive_rate_difference_t = 0

    # ****** 1. false_positive_rate_difference ******  optimal = 0
    r""":math:`FPR_{D = \text{unprivileged}} - FPR_{D = \text{privileged}}`
    """
    false_positive_rate_difference = calcul_simple_metric(group_values, "FPR", "diff")
    false_positive_rate_difference_t = 0

    # ****** 2. false_negative_rate_diff ******  optimal = 0
    r""":math:`FNR_{D = \text{unprivileged}} - FNR_{D = \text{privileged}}`
    """
    false_negative_rate_difference = calcul_simple_metric(group_values, "FNR", "diff")
    false_negative_rate_difference_t = 0

    # ****** 3. false_omission_rate_diff ******  optimal = 0
    r""":math:`FOR_{D = \text{unprivileged}} - FOR_{D = \text{privileged}}`
    """
    false_omission_rate_difference = calcul_simple_metric(group_values, "FOR", "diff")
    false_omission_rate_difference_t = 0

    # ****** 4. false_discovery_rate_diff ******  optimal = 0
    r""":math:`FDR_{D = \text{unprivileged}} - FDR_{D = \text{privileged}}`
    """
    false_discovery_rate_difference = calcul_simple_metric(group_values, "FDR", "diff")
    false_discovery_rate_difference_t = 0

    # ****** 5. false_positive_rate_ratio ******  optimal = 0
    r""":math:`\frac{FPR_{D = \text{unprivileged}}}{FPR_{D = \text{privileged}}}`
    """
    false_positive_rate_ratio = calcul_simple_metric(group_values, "FPR", "ratio")
    false_positive_rate_ratio_t = 1

    # ****** 6. false_negative_rate_ratio ******  optimal = 1
    r""":math:`\frac{FNR_{D = \text{unprivileged}}}{FNR_{D = \text{privileged}}}`
    """
    false_negative_rate_ratio = calcul_simple_metric(group_values, "FNR", "ratio")
    false_negative_rate_ratio_t = 1

    # ****** 7. false_omission_rate_ratio ******  optimal = 1
    r""":math:`\frac{FOR_{D = \text{unprivileged}}}{FOR_{D = \text{privileged}}}`
    """
    false_omission_rate_ratio = calcul_simple_metric(group_values, "FOR", "ratio")
    false_omission_rate_ratio_t = 1

    # ****** 8. false_discovery_rate_ratio ******  optimal = 1
    r""":math:`\frac{FDR_{D = \text{unprivileged}}}{FDR_{D = \text{privileged}}}`
    """
    false_discovery_rate_ratio = calcul_simple_metric(group_values, "FDR", "ratio")
    false_discovery_rate_ratio_t = 1

    # ****** 9. average_odds_difference ******  optimal = 0
    r"""Average of difference in FPR and TPR for unprivileged and privileged
    groups:

    .. math::

       \tfrac{1}{2}\left[(FPR_{D = \text{unprivileged}} - FPR_{D = \text{privileged}})
       + (TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}))\right]

    A value of 0 indicates equality of odds.
    """
    average_odds_difference = 0.5 * (calcul_simple_metric(group_values, "FPR", "diff")
                                     + calcul_simple_metric(group_values, "TPR", "diff"))
    average_odds_difference_t = 0

    # ****** 10. average_abs_odds_difference ******  optimal = 0
    r"""Average of absolute difference in FPR and TPR for unprivileged and
    privileged groups:

    .. math::

       \tfrac{1}{2}\left[|FPR_{D = \text{unprivileged}} - FPR_{D = \text{privileged}}|
       + |TPR_{D = \text{unprivileged}} - TPR_{D = \text{privileged}}|\right]

    A value of 0 indicates equality of odds.
    """
    average_abs_odds_difference = 0.5 * (np.abs(calcul_simple_metric(group_values, "FPR", "diff"))
                                     + np.abs(calcul_simple_metric(group_values, "TPR", "diff")))
    average_abs_odds_difference_t = 3

    # ****** 11. average_predictive_value_difference ******  optimal = 0
    r"""Average of difference in PPV and FOR for unprivileged and privileged
    groups:

    .. math::

       \tfrac{1}{2}\left[(PPV_{D = \text{unprivileged}} - PPV_{D = \text{privileged}})
       + (FOR_{D = \text{unprivileged}} - FOR_{D = \text{privileged}}))\right]

    A value of 0 indicates equality of chance of success.
    """
    average_predictive_value_difference = 0.5 * (calcul_simple_metric(group_values, "PPV", "diff") +
                                                 calcul_simple_metric(group_values, "FOR", "diff"))
    average_predictive_value_difference_t = 0

    # ****** 12. error_rate_difference ******  optimal = 0
    r"""Difference in error rates for unprivileged and privileged groups,
    :math:`ERR_{D = \text{unprivileged}} - ERR_{D = \text{privileged}}`.
    """
    error_rate_difference = calcul_simple_metric(group_values, "ERR", "diff")
    error_rate_difference_t = 0

    # ****** 13. error_rate_ratio ******  optimal = 1
    r"""Ratio of error rates for unprivileged and privileged groups,
    :math:`\frac{ERR_{D = \text{unprivileged}}}{ERR_{D = \text{privileged}}}`.
    """
    error_rate_ratio = calcul_simple_metric(group_values, "ERR", "ratio")
    error_rate_ratio_t = 1

    # selection_rate
    # There is no fairness metric called "selection_rate"

    # ****** 14. disparate_impact ******  optimal = 1
    r"""
    .. math::
       \frac{Pr(\hat{Y} = 1 | D = \text{unprivileged})}
       {Pr(\hat{Y} = 1 | D = \text{privileged})}
    """
    disparate_impact = calcul_simple_metric(group_values, "Selection_rate", "ratio")
    disparate_impact_t = 1

    # ****** 15. statistical_parity_difference ******  optimal = 1
    r"""
    .. math::
       Pr(\hat{Y} = 1 | D = \text{unprivileged})
       - Pr(\hat{Y} = 1 | D = \text{privileged})
    """
    statistical_parity_difference = calcul_simple_metric(group_values, "Selection_rate", "diff")
    statistical_parity_difference_t = 0

    # ****** 16. generalized_entropy_index ******  optimal = 0
    r"""Generalized entropy index is proposed as a unified individual and
    group fairness measure in [3]_.  With :math:`b_i = \hat{y}_i - y_i + 1`:

    .. math::

       \mathcal{E}(\alpha) = \begin{cases}
           \frac{1}{n \alpha (\alpha-1)}\sum_{i=1}^n\left[\left(\frac{b_i}{\mu}\right)^\alpha - 1\right],& \alpha \ne 0, 1,\\
           \frac{1}{n}\sum_{i=1}^n\frac{b_{i}}{\mu}\ln\frac{b_{i}}{\mu},& \alpha=1,\\
           -\frac{1}{n}\sum_{i=1}^n\ln\frac{b_{i}}{\mu},& \alpha=0.
       \end{cases}

    Args:
        alpha (int): Parameter that regulates the weight given to distances
            between values at different parts of the distribution.

    References:
        .. [3] T. Speicher, H. Heidari, N. Grgic-Hlaca, K. P. Gummadi, A. Singla, A. Weller, and M. B. Zafar,
           "A Unified Approach to Quantifying Algorithmic Unfairness: Measuring Individual and Group Unfairness via Inequality Indices,"
           ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2018.
    """
    benefits = pred_label - truelabel + 1
    generalized_entropy_index_val = generalized_entropy_index(benefits, 2)
    generalized_entropy_index_val_t = 3

    # ****** 17. betweeen_all_groups_generalized_entropy_index ******  optimal = 0
    # consider all groups, not only privil and unprivil
    """Between-group generalized entropy index that uses all combinations of
    groups based on `self.dataset.protected_attributes`. See
    :meth:`_between_group_generalized_entropy_index`.

    Args:
        alpha (int): See :meth:`generalized_entropy_index`.
    """
    betweeen_all_groups_generalized_entropy_index = _between_group_generalized_entropy_index(logits, truelabel, data,
                                                                                             sensitive_attributions,
                                                                                             alpha=2)
    betweeen_all_groups_generalized_entropy_index = max(betweeen_all_groups_generalized_entropy_index, 0)
    betweeen_all_groups_generalized_entropy_index_t = 3

    # ****** 18. betweeen_group_generalized_entropy_index ******  optimal = 0
    # just consider privil and unprivil
    """Between-group generalized entropy index that uses
    `self.privileged_groups` and `self.unprivileged_groups` as the only two
    groups. See :meth:`_between_group_generalized_entropy_index`.

    Args:
        alpha (int): See :meth:`generalized_entropy_index`.
    """
    betweeen_group_generalized_entropy_index = generalized_entropy_index(benefits_group, alpha=2)
    betweeen_group_generalized_entropy_index_t = 3

    # ****** 19. theil_index ******  optimal = 0
    r"""The Theil index is the :meth:`generalized_entropy_index` with
    :math:`\alpha = 1`.
    """
    theil_index = generalized_entropy_index(benefits, alpha=1)
    theil_index = max(theil_index, 0)
    theil_index_t = 3

    # ****** 20. coefficient_of_variation ******  optimal = 0
    r"""The coefficient of variation is the square root of two times the
    :meth:`generalized_entropy_index` with :math:`\alpha = 2`.
    """
    temp = generalized_entropy_index(benefits, alpha=2)
    coefficient_of_variation = np.sqrt(2 * max(temp, 0))
    coefficient_of_variation_t = 3

    # ****** 21. between_group_theil_index ******  optimal = 0
    r"""The between-group Theil index is the
    :meth:`between_group_generalized_entropy_index` with :math:`\alpha = 1`.
    """
    between_group_theil_index = generalized_entropy_index(benefits_group, alpha=1)
    between_group_theil_index = max(between_group_theil_index, 0)
    between_group_theil_index_t = 3

    # ****** 22. between_group_coefficient_of_variation ******  optimal = 0
    r"""The between-group coefficient of variation is the square
    root of two times the :meth:`between_group_generalized_entropy_index` with
    :math:`\alpha = 2`.
    """
    temp = generalized_entropy_index(benefits_group, alpha=2)
    between_group_coefficient_of_variation = np.sqrt(2 * max(temp, 0))
    between_group_coefficient_of_variation_t = 3

    # ****** 23. between_all_groups_theil_index ******  optimal = 0
    r"""The between-group Theil index is the
    :meth:`between_all_groups_generalized_entropy_index` with
    :math:`\alpha = 1`.
    """
    between_all_groups_theil_index = _between_group_generalized_entropy_index(logits, truelabel, data,
                                                                              sensitive_attributions,
                                                                              alpha=1)
    between_all_groups_theil_index_t = 3

    # ****** 24. between_all_groups_coefficient_of_variation ******  optimal = 0
    r"""The between-group coefficient of variation is the square
    root of two times the :meth:`between_group_generalized_entropy_index` with
    :math:`\alpha = 2`.
    """
    between_all_groups_coefficient_of_variation = np.sqrt(2 * betweeen_all_groups_generalized_entropy_index)
    between_all_groups_coefficient_of_variation_t = 3

    # ****** 25. differential_fairness_bias_amplification ******  optimal = 0
    # ####################### edf_clf
    concentration = 1.0
    num_classes = 2  # binary label dataset
    dirichlet_alpha = concentration / num_classes
    # compute counts for all intersecting groups, e.g. black-women, white-man, etc
    num_intersects = 2
    counts_pos = np.zeros(num_intersects)
    counts_total = np.zeros(num_intersects)
    for i in range(num_intersects):
        counts_total[i] = group_values[i]["totla_num"]
        counts_pos[i] = group_values[i]["TP"] + group_values[i]["FP"]

    # probability of y given S (p(y=1|S))
    ssr = (counts_pos + dirichlet_alpha) / (counts_total + concentration)

    # ####################
    def pos_ratio1(i, j):
        return abs(np.log(ssr[i]) - np.log(ssr[j]))

    def neg_ratio1(i, j):
        return abs(np.log(1 - ssr[i]) - np.log(1 - ssr[j]))

    edf_clf = max(max(pos_ratio1(i, j), neg_ratio1(i, j))
                  for i in range(len(ssr)) for j in range(len(ssr)) if i != j)
    # ####################### edf_data
    num_classes = 2  # binary label dataset
    dirichlet_alpha = concentration / num_classes
    num_intersects = 2
    counts_pos = np.zeros(num_intersects)
    counts_total = np.zeros(num_intersects)
    for i in range(num_intersects):
        counts_total[i] = group_values[i]["totla_num"]
        counts_pos[i] = group_values[i]["P"]

    # probability of y given S (p(y=1|S))
    sbr = (counts_pos + dirichlet_alpha) / (counts_total + concentration)

    def pos_ratio2(i, j):
        return abs(np.log(sbr[i]) - np.log(sbr[j]))

    def neg_ratio2(i, j):
        return abs(np.log(1 - sbr[i]) - np.log(1 - sbr[j]))

    edf_data = max(max(pos_ratio2(i, j), neg_ratio2(i, j))
                   for i in range(len(sbr)) for j in range(len(sbr)) if i != j)

    differential_fairness_bias_amplification = edf_clf - edf_data
    differential_fairness_bias_amplification_t = 0

    # ****** 26. average_violation of Dwork et al.'s pairwise constraints  optimal = 0
    r""" Fairness Behind a Veil of Ignorance: A Welfare Analysis for Automated Decision Making"""
    if data_dist is None:
        average_violation_Dwork = 0
    else:
        average_violation_Dwork = calculate_similar_dist(data_dist, pred_label)
    average_violation_Dwork_t = 3

    # ****** 27. accuracy ******  optimal = 1
    accuracy = np.mean(pred_label == truelabel)
    accuracy_t = 2

    # ****** 28. BCE_loss ******  optimal = 0
    BCE_loss = log_loss(truelabel.reshape(-1, 1), np.hstack([1 - logits.reshape(-1, 1), logits.reshape(-1, 1)]))
    BCE_loss_t = 3


    # record all the information
    Metric_info = dict(
        accuracy = accuracy,
        BCE_loss=BCE_loss,
        true_positive_rate_difference=true_positive_rate_difference,
        false_positive_rate_difference=false_positive_rate_difference,
        false_negative_rate_difference=false_negative_rate_difference,
        false_omission_rate_difference=false_omission_rate_difference,
        false_discovery_rate_difference=false_discovery_rate_difference,
        false_positive_rate_ratio=false_positive_rate_ratio,
        false_negative_rate_ratio=false_negative_rate_ratio,
        false_omission_rate_ratio=false_omission_rate_ratio,
        false_discovery_rate_ratio=false_discovery_rate_ratio,
        average_odds_difference=average_odds_difference,
        average_abs_odds_difference=average_abs_odds_difference,
        average_predictive_value_difference=average_predictive_value_difference,
        error_rate_difference=error_rate_difference,
        error_rate_ratio=error_rate_ratio,
        disparate_impact=disparate_impact,
        statistical_parity_difference=statistical_parity_difference,
        generalized_entropy_index=generalized_entropy_index_val,
        betweeen_all_groups_generalized_entropy_index=betweeen_all_groups_generalized_entropy_index,
        betweeen_group_generalized_entropy_index=betweeen_group_generalized_entropy_index,
        theil_index=theil_index,
        coefficient_of_variation=coefficient_of_variation,
        between_group_theil_index=between_group_theil_index,
        between_group_coefficient_of_variation=between_group_coefficient_of_variation,
        between_all_groups_theil_index=between_all_groups_theil_index,
        between_all_groups_coefficient_of_variation=between_all_groups_coefficient_of_variation,
        differential_fairness_bias_amplification=differential_fairness_bias_amplification,
        average_violation_Dwork=average_violation_Dwork
    )
    Objective_t_info = dict(
        accuracy=accuracy_t,
        BCE_loss=BCE_loss_t,
        true_positive_rate_difference=true_positive_rate_difference_t,
        false_positive_rate_difference=false_positive_rate_difference_t,
        false_negative_rate_difference=false_negative_rate_difference_t,
        false_omission_rate_difference=false_omission_rate_difference_t,
        false_discovery_rate_difference=false_discovery_rate_difference_t,
        false_positive_rate_ratio=false_positive_rate_ratio_t,
        false_negative_rate_ratio=false_negative_rate_ratio_t,
        false_omission_rate_ratio=false_omission_rate_ratio_t,
        false_discovery_rate_ratio=false_discovery_rate_ratio_t,
        average_odds_difference=average_odds_difference_t,
        average_abs_odds_difference=average_abs_odds_difference_t,
        average_predictive_value_difference=average_predictive_value_difference_t,
        error_rate_difference=error_rate_difference_t,
        error_rate_ratio=error_rate_ratio_t,
        disparate_impact=disparate_impact_t,
        statistical_parity_difference=statistical_parity_difference_t,
        generalized_entropy_index=generalized_entropy_index_val_t,
        betweeen_all_groups_generalized_entropy_index=betweeen_all_groups_generalized_entropy_index_t,
        betweeen_group_generalized_entropy_index=betweeen_group_generalized_entropy_index_t,
        theil_index=theil_index_t,
        coefficient_of_variation=coefficient_of_variation_t,
        between_group_theil_index=between_group_theil_index_t,
        between_group_coefficient_of_variation=between_group_coefficient_of_variation_t,
        between_all_groups_theil_index=between_all_groups_theil_index_t,
        between_all_groups_coefficient_of_variation=between_all_groups_coefficient_of_variation_t,
        differential_fairness_bias_amplification=differential_fairness_bias_amplification_t,
        average_violation_Dwork=average_violation_Dwork_t
    )
    return Metric_info, Objective_t_info


def calculate_similar_dist(data_dist, y):
    data_dist = data_dist.reshape(1, -1)
    data_dist = data_dist / np.max(data_dist)
    y = y.reshape(1, -1)

    y_diff = pdist(y.T, 'cityblock')

    Dwork_value = y_diff - data_dist

    flag = Dwork_value < 0
    Dwork_value[flag] = 0

    Dwork_value = np.mean(Dwork_value)
    return Dwork_value


def _between_group_generalized_entropy_index(logits, truelabel, data, sensitive_attributions, alpha=2):
    # consider all groups, not only privil and unprivil
    logits = logits.reshape(1, -1).astype(np.float64)
    truelabel = truelabel.astype(np.float64).reshape(1, -1)
    pred_label = get_label(logits.copy())

    benefits = logits - truelabel + 1  # new version in section 3.1
    b = copy.deepcopy(benefits)
    b[0, :] = 0

    attribution = data.columns
    group_attr = []
    group_dict = {}

    for sens in sensitive_attributions:
        temp = []
        for attr in attribution:
            temp1 = sens + '_'
            if temp1 in attr:
                temp.append(attr)
        group_dict.update({sens: temp})

    for sens in sensitive_attributions:
        group_attr.append(group_dict[sens])

    for item in product(*eval(str(group_attr))):
        group = item
        flag = np.ones([1, truelabel.shape[1]]) == np.ones([1, truelabel.shape[1]])
        for g in group:
            flag = flag & data[g]
        g_num = np.sum(flag)
        if g_num != 0:
            g_idx = np.array(np.where(flag)).reshape([1, -1])[0]
            g_truelabel = truelabel[0, g_idx]
            g_predlabel = pred_label[0, g_idx]
            b[0, g_idx] = np.mean(g_predlabel - g_truelabel + 1)

    if alpha == 1:
        return np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b))
    elif alpha == 0:
        return -np.mean(np.log(b / np.mean(b)) / np.mean(b))
    else:
        return np.mean((b / np.mean(b)) ** alpha - 1) / (alpha * (alpha - 1))


def generalized_entropy_index(b, alpha):
    # https://github.com/Trusted-AI/AIF360/blob/master/aif360/metrics/classification_metric.py#L664
    # benefits = pred_label - truelabel + 1  # original version
    if alpha == 1:
        return np.mean(np.log((b / np.mean(b)) ** b) / np.mean(b))
    elif alpha == 0:
        return -np.mean(np.log(b / np.mean(b)) / np.mean(b))
    else:
        return np.mean((b / np.mean(b)) ** alpha - 1) / (alpha * (alpha - 1))


def _smoothed_base_rates(labels, concentration=1.0):
    """Dirichlet-smoothed base rates for each intersecting group in the
    dataset.
    """
    # Dirichlet smoothing parameters
    if concentration < 0:
        raise ValueError("Concentration parameter must be non-negative.")
    num_classes = 2  # binary label dataset
    dirichlet_alpha = concentration / num_classes

    # compute counts for all intersecting groups, e.g. black-women, white-man, etc
    intersect_groups = np.unique(self.dataset.protected_attributes, axis=0)
    num_intersects = len(intersect_groups)
    counts_pos = np.zeros(num_intersects)
    counts_total = np.zeros(num_intersects)
    for i in range(num_intersects):
        condition = [dict(zip(self.dataset.protected_attribute_names,
                              intersect_groups[i]))]
        counts_total[i] = utils.compute_num_instances(
            self.dataset.protected_attributes,
            self.dataset.instance_weights,
            self.dataset.protected_attribute_names, condition=condition)
        counts_pos[i] = utils.compute_num_pos_neg(
            self.dataset.protected_attributes, labels,
            self.dataset.instance_weights,
            self.dataset.protected_attribute_names,
            self.dataset.favorable_label, condition=condition)

    # probability of y given S (p(y=1|S))
    return (counts_pos + dirichlet_alpha) / (counts_total + concentration)


def Cal_objectives(data, data_norm, logits, truelabel, sensitive_attributions, alpha, obj_names=None,
                   privileged_class_names=None, data_dist=None):
    sum_num = logits.shape[0] * logits.shape[1]
    logits = np.array(logits).reshape([1, sum_num])
    try:
        truelabel = truelabel.detach().cpu().numpy().reshape([1, sum_num])
    except:
        truelabel = truelabel.reshape([1, sum_num])
    Metric_info, Objective_t_info = calcul_all_fairness_objs_new(data,
                                               data_norm,
                                               logits,
                                               truelabel,
                                               sensitive_attributions,
                                               alpha, obj_names,
                                               privileged_class_names,
                                               data_dist)
    return Metric_info, Objective_t_info
