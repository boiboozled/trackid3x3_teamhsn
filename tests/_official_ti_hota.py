"""Copy of the official TI-HOTA functions, code verbatim, LaTeX escapes stripped from docstrings (cell 0 of
https://github.com/open-starlab/TrackID3x3/blob/9822a3d/script/compute_TI-HOTA.ipynb),
Apache-2.0, (c) the TrackID3x3 authors. Used only as the reference in tests."""
# fmt: off
# ruff: noqa
import numpy as np
import math
from scipy.optimize import linear_sum_assignment

# Calculation of location similarity LocSim
def loc_sim(P_pos, G_pos, tau):
    """
    Function to compute location similarity LocSim
    
    [
    operatorname{LocSim}(P, G) = exp( ln(0.05) * frac{|P - G|_2^2}{tau^2} )
    ]
    
    :param P_pos: Predicted position (coordinates in list or numpy array)
    :param G_pos: Ground Truth Location
    :param tau: normalized parameter
    :return: LocSim Value
    """
    P_pos = np.array(P_pos)
    G_pos = np.array(G_pos)
    distance_sq = np.sum((P_pos - G_pos)**2)
    sim = math.exp(math.log(0.05) * (distance_sq / (tau**2)))
    return sim

# Calculation of attribute similarity IdSim
def id_sim(P_attr, G_attr):
    """
    Function to calculate attribute similarity IdSim
    [
    operatorname{IdSim}(P, G) =
    begin{cases}
    1, & text{if } P_{text{attr}} = G_{text{attr}}, 
    0, & text{otherwise.}
    end{cases}
    ]
    :param P_attr: Attributes of prediction (specify each attribute in a dictionary, etc.)
    :param G_attr: Ground Truth Attributes
    :return: 1 if all attributes match, 0 otherwise
    """

    return 1 if P_attr == G_attr else 0

# Overall similarity of each pair
def sim_ti_hota(P, G, tau):
    """
    Function to calculate the overall similarity of TI-HOTA
    [
    operatorname{Sim}_{mathrm{TItext{-}HOTA}}(P, G) = operatorname{LocSim}(P, G) x operatorname{IdSim}(P, G)
    ]
    :param P: Prediction Information {'position': [x, y, ...], 'attributes': {...}} 
    :param G: Ground Truth Information {'position': [x, y, ...], 'attributes': {...}} 
    :param tau: normalized parameter for loc_sim calculation
    :return: overall similarity Sim_{TI-HOTA}
    """
    
    return loc_sim(P['position'], G['position'], tau) * id_sim(P['attributes'], G['attributes'])

def compute_ti_hota_alpha(frames, tau, alpha):
    total_tp = total_fp = total_fn = 0
    assoc_counts = {}
    fpa_counts = {}

    for frame in frames:
        gt_objs = frame.get('ground_truths', [])
        pred_objs = frame.get('predictions', [])
        n_gt, n_pred = len(gt_objs), len(pred_objs)
        
        # Build similarity and cost matrices
        combined = np.zeros((n_gt, n_pred))
        cost = np.zeros((n_gt, n_pred))
        for i, gt in enumerate(gt_objs):
            for j, pred in enumerate(pred_objs):
                loc = loc_sim(pred['position'], gt['position'], tau)
                combined[i, j] = loc * id_sim(pred['attributes'], gt['attributes'])
                cost[i, j] = 1 - combined[i, j]

        # Exclude below-threshold pairs
        cost[combined < alpha] = 1e6
        row_ind, col_ind = linear_sum_assignment(cost)

        # Determine current valid matches
        curr_gt_match = {}
        for i, j in zip(row_ind, col_ind):
            if combined[i, j] >= alpha:
                curr_gt_match[i] = j

        # Update detection counts
        tp = len(curr_gt_match)
        fp = n_pred - tp
        fn = n_gt   - tp
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Association counts (for AssA calculation)
        matched_gt = set(curr_gt_match.keys())
        for i, gt in enumerate(gt_objs):
            gid = gt['attributes']
            assoc_counts.setdefault(gid, {'TPA': 0, 'FNA': 0})
            if i not in matched_gt:
                assoc_counts[gid]['FNA'] += 1
        for i, j in curr_gt_match.items():
            gid = gt_objs[i]['attributes']
            assoc_counts[gid]['TPA'] += 1
        for j, pred in enumerate(pred_objs):
            if j not in curr_gt_match.values():
                pid = pred['attributes']
                fpa_counts[pid] = fpa_counts.get(pid, 0) + 1

    # Compute average DetA and AssA
    total_det = total_tp + total_fp + total_fn
    det_acc = total_tp / total_det if total_det > 0 else 0
    sum_A = n_tracks = 0
    for gid, cnt in assoc_counts.items():
        denom = cnt['TPA'] + cnt['FNA'] + fpa_counts.get(gid, 0)
        if denom > 0:
            sum_A += cnt['TPA'] / denom
            n_tracks += 1
    ass_acc = sum_A / n_tracks if n_tracks > 0 else 0

    # TI-HOTA for this alpha
    ti_hota_alpha = math.sqrt(det_acc * ass_acc)
    
    return ti_hota_alpha, det_acc, ass_acc, total_tp, total_fp, total_fn

# The function that calculates TI-HOTA for each threshold value α and takes the mean of the TI-HOTA_α
def compute_ti_hota(frames, tau):
    alphas = np.arange(0.05, 1.0, 0.05)
    ti_hota_alphas = []
    det_acc_values = []
    ass_acc_values = []
    total_tp_values = []
    total_fp_values = []
    total_fn_values = []
    
    for alpha in alphas:
        ti_hota_alpha, det_acc, ass_acc, tp, fp, fn = compute_ti_hota_alpha(frames, tau, alpha)
        ti_hota_alphas.append(ti_hota_alpha)
        det_acc_values.append(det_acc)
        ass_acc_values.append(ass_acc)
        total_tp_values.append(tp)
        total_fp_values.append(fp)
        total_fn_values.append(fn)
        
    ti_hota = np.mean(ti_hota_alphas)
    mean_det_acc = np.mean(det_acc_values)
    mean_ass_acc = np.mean(ass_acc_values)
    mean_total_tp = np.mean(total_tp_values)
    mean_total_fp = np.mean(total_fp_values)
    mean_total_fn = np.mean(total_fn_values)
    
    metrics = {
        "TI-HOTA": ti_hota,
        "TI-HOTA_alphas": ti_hota_alphas,
        "TI-DetA": mean_det_acc,
        "TI-AssA": mean_ass_acc,
        "TI-TP": mean_total_tp,
        "TI-FP": mean_total_fp,
        "TI-FN": mean_total_fn,
    }
    return metrics, alphas
