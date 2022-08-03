import torch
import numpy as np


def FPR_AT_95_TPR(TPR, FPR):
    """  Measures the false positive rate when the true positive rate is equal to 95%
        TPR -> list: True Positive Rate
        FPR -> list: False Positive Rate
    return:
        fpr_at_95_tpr -> float: the fpr_at_95_tpr
    """
    for i in range(len(TPR)):
        if 0.9505 >= TPR[i] >= 0.9495:
            return FPR[i]
    return 0


def ece(results, precision=15):
    """ Expected Calibration Error (ECE)
        results   -> tensor: (uncertainty, prediction, labels), the tensor has to be sorted by uncertainty
        precision -> int: numbers of bins
    return:
        ece_score -> float: the ece score
        tab_conf  -> list: the list for each bin of the ocnfidence score
        tab_acc   -> list: the list for each bin of the accuracy score
    """
    res = results[:, 0] / torch.max(results[:, 0])
    bin_boundaries = torch.linspace(0, 1, precision + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidences = 1 - res
    accuracies = results[:, -1].eq(results[:, -2])
    tab_acc, tab_conf = [], []
    ece_score = torch.zeros(1, device=results.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece_score += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            tab_acc.append(accuracy_in_bin)
            tab_conf.append(avg_confidence_in_bin)
    return ece_score.item(), tab_conf, tab_acc


def ace(results, precision):
    """ Adaptive Calibration Error (ACE)
        results -> tensor: (uncertainty, prediction, labels), the tensor has to be sorted by uncertainty
        precision -> int: numbers of bins
    return:
        ace_score -> float: the ace score
    """

    results[:, 0] /= torch.max(results[:, 0])                                    # Standardize the input between 0 and 1
    ace_score = torch.zeros(1, device=results.device)
    nb_classes = torch.unique(results[:, -1])                                    # All possible classes
    for k in nb_classes:
        res_k = results[results[:, -1] == k]                                     # Select only the classe k
        acc_k = res_k[:, -1].eq(res_k[:, -2])
        conf_k = 1 - res_k[:, 0]                                                 # Convert uncertainty to confidence
        bin_boundaries = np.linspace(0, len(res_k), precision, dtype=int)
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            accuracy_in_bin = acc_k[bin_lower:bin_upper].float().mean()
            avg_confidence_in_bin = conf_k[bin_lower:bin_upper].float().mean()
            ace_score += torch.abs(accuracy_in_bin - avg_confidence_in_bin)
    ace_score /= (len(nb_classes) * precision)
    return ace_score.item()


def compute_score(tab, args):
    """ Compute different kind of metrics to evaluate OOD detection
        tab  -> tensor: (uncertainty, prediction, labels), the prediction with the confidence score
        args ->  Argparse: global arguments
    return:
        acc          -> the accuracy
        aupr_success -> area under the precison-recall curve where positive class are correct predictions
        aupr_error   -> area under the precison-recall curve where positive class are errors
        auroc        -> area under the roc curve
        fpr_at_95tpr -> false positive rate when the true positive rate is equal to 95%
        ace_score    -> Adaptive Calibration Error
        ece_score    -> Expected Calibration Error
    """

    tab = tab[tab[:, 0].argsort()]
    tab = tab.to(args.device)

    TPR, FPR = [], []
    list_P_sucess, list_R_sucess = [], []
    list_P_error, list_R_error = [], []
    for ind, i in enumerate(torch.linspace(0, len(tab), 10_000)):
        if ind == 0:
            continue
        i = int(i)
        TP = torch.where(tab[:i, -2] == tab[:i, -1], args.one, args.zero).sum().cpu().item()
        FP = torch.where(tab[:i, -2] != tab[:i, -1], args.one, args.zero).sum().cpu().item()
        FN = torch.where(tab[i:, -2] == tab[i:, -1], args.one, args.zero).sum().cpu().item()
        TN = torch.where(tab[i:, -2] != tab[i:, -1], args.one, args.zero).sum().cpu().item()

        TPR.append(TP / (TP + FN + 1e-11))
        FPR.append(FP / (FP + TN + 1e-11))
        list_P_sucess.append(TP / (TP + FP + 1e-11))
        list_R_sucess.append(TP / (TP + FN + 1e-11))
        list_P_error.insert(0, TN / (TN + FN + 1e-11))
        list_R_error.insert(0, TN / (TN + FP + 1e-11))

    #acc = tab[:, -2].eq(tab[:, -1]).cpu().numpy().mean()
    aupr_success = np.trapz(list_P_sucess, list_R_sucess)
    aupr_error = np.trapz(list_P_error, list_R_error)
    auroc = np.trapz(TPR, FPR)
    fpr_at_95tpr = FPR_AT_95_TPR(TPR, FPR)
    ace_score = ace(tab.cpu(), 15)

    return [aupr_success, aupr_error, auroc, fpr_at_95tpr, ace_score]


def print_result(name, split, res, writer, epoch, args):
    """ Print and return the evaluation of the prediction
        name      -> str: name of the method
        split     -> str: either Train, Val or Test
        res       -> tensor: (uncertainty, prediction, labels), the prediction with the confidence score
        writer    -> SummaryWriter: for tensorboard logs
        epoch     -> int: current epoch
        args      -> Argparse: global arguments
    return:
        results -> dict: the all the metrics computed
    """

    aupr_success, aupr_error, auroc, fpr_at_95tpr, ace = compute_score(res, args)

    s = "\r" + split + " " + name
    s += ': Fpr_at_95tpr {:.1f}, Aupr sucess: {:.1f}, Aupr error: {:.1f}, Roc: {:.1f},  Ace {:.3f}'
    print(s.format(fpr_at_95tpr * 100, aupr_success * 100, aupr_error * 100, auroc * 100, ace))
    writer.add_scalars('data/' + split + '_metric_' + name,
                       {"Aupr sucess": aupr_success, "Aupr error": aupr_error,
                        "Auroc": auroc, "fpr_at_95tpr": fpr_at_95tpr, "ace": ace},
                       epoch)

    results = {"auroc": auroc, "aupr": aupr_success, "fpr_at_95tpr": fpr_at_95tpr, "ace": ace, "ece": ece}
    return results
