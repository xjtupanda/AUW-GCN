import torch
import numpy as np
import pandas as pd
import os

def _cal_proposal(opt, array_softmax_score, array_apex_score, offset, vid_name, exp):
    array_score_micro_start = array_softmax_score[:, 0, :]
    array_score_micro_end = array_softmax_score[:, 1, :]
    proposal_block = calculate_proposal_with_score(opt, 
        array_score_micro_start, array_score_micro_end,
        array_apex_score, offset, vid_name, exp)
    
    return proposal_block


def calculate_proposal_with_score(opt, array_score_start, array_score_end,
                                  array_score_apex, offset, vid_name, exp):
    ret = []
    if exp == 2:        # micro-expression
        left_min_dis = opt['micro_left_min_dis']
        left_max_dis = opt['micro_left_max_dis']
        right_min_dis = opt['micro_right_min_dis']
        right_max_dis = opt['micro_right_max_dis']
        min_len = opt['micro_min']
        max_len = opt['micro_max']
        # STEP = int(opt["RECEPTIVE_FILED"] // 2)  # int(opt["micro_average_len"]*3/2)  == 7
        # EX_MIN = int(opt["micro_min"] // 2)                                           == 2
        apex_score_threshold = opt["micro_apex_score_threshold"]
    elif exp == 1:      # macro-expression
        return ret # for demo. only display ME
        # STEP = int(opt["macro_average_len"] // 2)  # int(opt["micro_average_len"]*3/2)
        # EX_MIN = int(opt["macro_min"] // 2)
        left_min_dis = opt['macro_left_min_dis']
        left_max_dis = opt['macro_left_max_dis']
        right_min_dis = opt['macro_right_min_dis']
        right_max_dis = opt['macro_right_max_dis']
        min_len = opt['macro_min']
        max_len = int(opt['macro_average_len'] *3/2)
        apex_score_threshold = opt["macro_apex_score_threshold"]
    else:
        raise ValueError
    
    apex_indices = np.nonzero(array_score_apex > apex_score_threshold) # (#apex, 2=batch_index + window_index) : score: (B, T)
    batch_index, window_index = apex_indices   # coordinate for axis 0, axis 1
    k = len(batch_index)        # k = # of potential proposals; i.e., index where apex score is larger than threshold.
    if k == 0:
        return ret
    
    N, T = array_score_start.shape # N = # of windows = B, window_size
    
    # search in the left
    _tmp = np.arange(left_min_dis, left_max_dis + 1, dtype=np.int64).reshape(1, -1)       # search range, auxiliary array.
    M = _tmp.shape[1]
    start_indices = np.maximum(window_index.reshape(-1, 1) - _tmp, 0) # (K, 1) - (1, M) -> (K, M)
    tmp_start_indices = start_indices.reshape(-1)# (k * M=length search range)
    tmp_batch_index = np.repeat(batch_index, M).reshape(-1)           # expand to k * M to align with tmp_start_indices
    start_indices_indices = np.argmax(array_score_start[tmp_batch_index, tmp_start_indices].reshape(k, M),
                                        axis=-1)  # (k, 1)
    start_indices = start_indices[np.arange(start_indices.shape[0]),
                                    start_indices_indices]
    
    # search in the right
    _tmp = np.arange(right_min_dis, right_max_dis + 1, dtype=np.int64).reshape(1, -1)       # search range, auxiliary array.
    M = _tmp.shape[1]
    end_indices = np.minimum(window_index.reshape(-1, 1) + _tmp,
                                T - 1)
    tmp_end_indices = end_indices.reshape(-1)# (k * M=length search range)
    tmp_batch_index = np.repeat(batch_index, M).reshape(-1)
    end_indices_indices = np.argmax(array_score_end[tmp_batch_index, tmp_end_indices].reshape(k, M), 
                                    axis=-1)
    end_indices = end_indices[np.arange(end_indices.shape[0]),
                                end_indices_indices]
    
    #start_indices: (k), end_indices: (k) k = # of potential proposals
    for idx in range(k):
        start_index = start_indices[idx].item()
        end_index = end_indices[idx].item()
        apex_index = window_index[idx].item()
        x_index = batch_index[idx].item()
        if (array_score_start[x_index, start_index] > array_score_end[x_index, start_index]) \
        and(array_score_start[x_index, end_index]   < array_score_end[x_index, end_index]):
            _offset = offset[x_index].item()
            _vid_name = vid_name[x_index]
            tmp_len = end_index - start_index
            if tmp_len < min_len or tmp_len > max_len:
                continue
            st = start_index + _offset + 1  # add 1 due to transformation from zero-indexed
            ed = end_index + _offset + 1    # add 1 due to transformation from zero-indexed
            st_score = array_score_start[x_index, start_index]
            ed_score = array_score_end[x_index, end_index]
            ap_score = array_score_apex[x_index, apex_index]
            # col_name = ["video_name", "start_frame", "end_frame", "start_socre",
            #     "end_score", "apex_score", "type_idx"]
            ret.append([_vid_name, st, ed, st_score, ed_score, ap_score, exp]) # final score = s*ap*ed
    
    if len(ret) > 0:
        ret = np.stack(ret, axis=0)
    return ret      # (#final proposals, 4)

def eval_single_epoch(opt, model, dataloader, epoch, device):
    print(f'Evaluating ckpt of epoch {epoch:03d}')
    model.eval()
    # vid_name: 'casme_016_0505',...
    # offset : tensor([2048, 1024, 2048, 3072, 2560,    0, 1536, 1280]) etc.
    # feature: (B, T=256(window_size) + 2*7(padding), N=12, 2=x+y)
    for (feature, offset, vid_name) in dataloader:
        feature = feature.to(device)
        
        # align with training, orig. approach is to use the complete feature without segmentation though.
        # TODO: use complete feature as orig.
        STEP = int(opt["RECEPTIVE_FILED"] // 2)

        output_probability = model(feature)
        output_probability = output_probability[:, :, STEP:-STEP]   # (B, T=256, C=24)

        output_micro_apex = output_probability[:, 6, :]
        output_macro_apex = output_probability[:, 7, :]
        output_micro_start_end = output_probability[:, 0: 0 + 3, :]
        output_macro_start_end = output_probability[:, 3: 3 + 3, :]

        tmp_array = []
        _COL_NAME = ["video_name", "start_frame", "end_frame", "start_score",
                "end_score", "apex_score", "type_idx"]
        # micro expression
        array_softmax_score_micro = torch.softmax(
            output_micro_start_end, dim=1).cpu().numpy()        # (B, 3, T)
        array_score_micro_apex = torch.sigmoid(                 
            output_micro_apex).cpu().numpy()                    # (B, T)
        
        micro_proposals = _cal_proposal(opt,
            array_softmax_score_micro, array_score_micro_apex, offset, vid_name,
            2) # 'micro' type==2
        if len(micro_proposals) > 0:
            tmp_array.append(micro_proposals)
        
        
        # macro expression
        array_softmax_score_macro = torch.softmax(
            output_macro_start_end, dim=1).cpu().numpy()        # (B, 3, T)
        array_score_macro_apex = torch.sigmoid(
            output_macro_apex).cpu().numpy()                    # (B, T)
        macro_proposals = _cal_proposal(opt,
            array_softmax_score_macro, array_score_macro_apex, offset, vid_name,
            1) # 'macro' type==1
        if len(macro_proposals) > 0:
            tmp_array.append(macro_proposals)
        
        # if there's proposal, save it in a .csv file
        if len(tmp_array) > 0:
            tmp_array = np.concatenate(tmp_array, axis=0)
            new_df = pd.DataFrame(tmp_array, columns=_COL_NAME)
            convert_dict = {"video_name":str, 
                            "start_frame":int, 
                            "end_frame":int, 
                            "start_score":float,
                            "end_score":float, 
                            "apex_score":float, 
                            "type_idx":int}
            new_df = new_df.astype(convert_dict)
            # TODO: the score should be a weighted sum
            new_df["score"] = new_df.start_score.values[:] * new_df.end_score.values[:] * new_df.apex_score.values[:]
            new_df = new_df.groupby('video_name').apply(
                lambda x: x.sort_values("score", ascending=False)).reset_index(drop=True)

            csv_dir = os.path.join(
                opt['output_dir_name'], 'output_csv'
            )
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
                
            predict_file = os.path.join(
                csv_dir, 'proposals_epoch_' + str(epoch).zfill(3) + '.csv'
            )
            
            if os.path.exists(predict_file):
                os.remove(predict_file)
            new_df.to_csv(predict_file, index=False)

def nms_single_epoch(opt, epoch):
    csv_dir = os.path.join(
                opt['output_dir_name'], 'output_csv'
    )
    predict_file = os.path.join(
                csv_dir, 'proposals_epoch_' + str(epoch).zfill(3) + '.csv'
    )
    # no proposals generated for this epoch
    if not os.path.exists(predict_file):
        return
    
    nms_dir = os.path.join(
        opt['output_dir_name'], 'nms_csv'
    )
    if not os.path.exists(nms_dir):
        os.makedirs(nms_dir)
        
    nms_file = os.path.join(
        nms_dir, 'final_proposals_epoch_' + str(epoch).zfill(3) + '.csv'
    )
    
    df = pd.read_csv(predict_file)
    df = df.groupby(['video_name', "type_idx"]).apply(
            lambda x: nms(x, opt)).reset_index(drop=True)
    
    if os.path.exists(nms_file):
        os.remove(nms_file)
    df.to_csv(nms_file, index=False)
    
def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def nms(df, opt):
    tstart = list(df.start_frame.values)
    tend = list(df.end_frame.values)
    tscore = list(df.score.values)
    video_name = df.video_name.values[0]
    type_idx = df.type_idx.values[0]

    if type_idx == 2:   # micro
        proposal_num = opt['nms_top_K_micro']
    elif type_idx == 1:
        proposal_num = opt['nms_top_K_macro']
    rstart = []
    rend = []
    rscore = []

    
    while len(tscore) > 0 and len(rscore) < proposal_num:
        max_index = np.argmax(tscore)
        if (tscore[max_index]) == 0:
            break
        iou_list = iou_with_anchors(
            tstart[max_index], tend[max_index],
            np.array(tstart), np.array(tend))
        # iou_exp_list = np.exp(-np.square(iou_list)/0.75)
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = iou_list[idx]
                if tmp_iou > 0.5:
                    tscore[idx] = 0  # tscore[idx]*iou_exp_list[idx]

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])

        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['video_name'] = [video_name] * len(rstart)
    newDf['start_frame'] = rstart
    newDf['end_frame'] = rend
    newDf['score'] = rscore
    newDf["type_idx"] = [type_idx] * len(rstart)
    return newDf


def iou_for_find(df, opt):
    video_name = df.video_name.values[0]
    type_idx = df.type_idx.values[0]

    anno_df = pd.read_csv(opt['anno_csv'])
    tmp_anno_df = anno_df[anno_df['video_name'] == video_name]
    tmp_anno_df = tmp_anno_df[tmp_anno_df['type_idx'] == type_idx]

    gt_start_indices = tmp_anno_df.start_frame.values // opt["RATIO_SCALE"]
    gt_end_indices = tmp_anno_df.end_frame.values // opt["RATIO_SCALE"]
    predict_start_indices = df.start_frame.values
    predict_end_indices = df.end_frame.values

    tiou = np.array([0.] * len(df))
    idx_list = []

    for j in range(len(gt_start_indices)):
        gt_start = gt_start_indices[j]
        gt_end = gt_end_indices[j]
        ious = iou_with_anchors(gt_start, gt_end, predict_start_indices, predict_end_indices)
        max_iou = max(ious)
        if max_iou > 0.5:
            tmp_idx = np.argmax(ious)
            idx_list.append(tmp_idx)
            tiou[tmp_idx] = max_iou

    find = np.array(["False"] * len(df))
    find[idx_list] = "True"
    df["find"] = find
    df["iou"] = tiou
    return df


def iou_for_tp(df, opt):
    video_name = df.video_name.values[0]
    type_idx = df.type_idx.values[0]

    anno_df = pd.read_csv(opt['anno_csv'])
    tmp_anno_df = anno_df[anno_df['video_name'] == video_name]
    tmp_anno_df = tmp_anno_df[tmp_anno_df['type_idx'] == type_idx]

    if len(tmp_anno_df) == 0:
        tp = np.array(["False"] * len(df))
        df["tp"] = tp
        df["iou"] = 0
        return df

    gt_start_indices = tmp_anno_df.start_frame.values // opt["RATIO_SCALE"]
    gt_end_indices = tmp_anno_df.end_frame.values // opt["RATIO_SCALE"]
    predict_start_indices = df.start_frame.values
    predict_end_indices = df.end_frame.values

    tiou = np.array([0.] * len(df))
    idx_list = []

    for j in range(len(predict_start_indices)):
        predict_start = predict_start_indices[j]
        predict_end = predict_end_indices[j]
        ious = iou_with_anchors(predict_start, predict_end, gt_start_indices, gt_end_indices)
        max_iou = max(ious)
        if max_iou > 0.5:
            tiou[j] = max_iou
            idx_list.append(j)

    tp_fp = np.array(["False"] * len(df))
    tp_fp[idx_list] = "True"
    df["tp"] = tp_fp
    df["iou"] = tiou
    return df

def calc_metrics(df, opt):
    
    tp = 0
    n = 0       # num of proposals
    tp += len(df[df["tp"] == "True"])
    n += len(df)
        
    return tp, n

def calculate_epoch_metrics(opt):
    
    nms_dir = os.path.join(
        opt['output_dir_name'], 'nms_csv'
    )
    def _cal_metrics(tp, n, m):
        recall = float(tp) / m if m > 0 else 0
        precision = float(tp) / n if n > 0 else 0
        f1_score = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        
        return recall, precision, f1_score
    
    # _COL_NAME = ['micro_precision', 'micro_recall', 'micro_f1',
    #              'macro_precision', 'macro_recall', 'macro_f1',
    #              'all_precision',   'all_recall',   'all_f1',
    #              'epoch']
    
    _COL_NAME = ['micro_precision', 'micro_recall', 'micro_f1',
                 'macro_precision', 'macro_recall', 'macro_f1',
                 'all_precision',   'all_recall',   'all_f1',
                 'micro_tp', 'micro_n', 'micro_m',
                 'macro_tp', 'macro_n', 'macro_m',
                 'all_tp',  'all_n',    'all_m',
                 'epoch']
    
    tmp_list = []
    
    epoch_begin = opt['epoch_begin']
    for epoch in range(epoch_begin, opt['epochs']):
        nms_file = os.path.join(
            nms_dir, 'final_proposals_epoch_' + str(epoch).zfill(3) + '.csv'
        )
        if not os.path.exists(nms_file):continue
        nms_df = pd.read_csv(nms_file)
        
        new_df = nms_df.groupby(['video_name', "type_idx"]).apply(
            lambda x: iou_for_find(x, opt)).reset_index(drop=True)
        new_df = new_df.groupby(['video_name', "type_idx"]).apply(
            lambda x: iou_for_tp(x, opt)).reset_index(drop=True)
        
        
        res = new_df.groupby(['type_idx']).apply(
            lambda x: calc_metrics(x, opt)).to_dict()
        
        
        mic_rec, mic_pr, mic_f1 = [0] * 3
        mac_rec, mac_pr, mac_f1 = [0] * 3
        all_rec, all_pr, all_f1 = [0] * 3
        micro_tp, micro_n, micro_m = [0] * 3
        macro_tp, macro_n, macro_m = [0] * 3
        all_tp, all_n, all_m = [0] * 3
        
        anno_df = pd.read_csv(opt['anno_csv'])
        tmp_df = anno_df[(anno_df['type_idx'] == 1 )
                     & (anno_df['subject'] == opt['subject'])]
        macro_m += len(tmp_df)
        
        tmp_df = anno_df[(anno_df['type_idx'] == 2 )
                     & (anno_df['subject'] == opt['subject'])]
        micro_m += len(tmp_df)
        
        for k, v in res.items():
            if k == 1: # macro
                tp, n = v
                macro_tp += tp
                macro_n += n
            elif k == 2: #micro
                tp, n = v
                micro_tp += tp
                micro_n += n
        
        mac_rec, mac_pr, mac_f1 = _cal_metrics(macro_tp, macro_n, macro_m)
        mic_rec, mic_pr, mic_f1 = _cal_metrics(micro_tp, micro_n, micro_m)       
        all_tp += micro_tp + macro_tp
        all_n += micro_n + macro_n
        all_m += micro_m + macro_m
        
        all_rec, all_pr, all_f1 = _cal_metrics(all_tp, all_n, all_m)
        
        # _COL_NAME = ['micro_precision', 'micro_recall', 'micro_f1',
        #          'macro_precision', 'macro_recall', 'macro_f1',
        #          'all_precision',   'all_recall',   'all_f1',
        #          'micro_tp', 'micro_n', 'micro_m',
        #          'macro_tp', 'macro_n', 'macro_m',
        #          'all_tp',  'all_n',    'all_m',
        #          'epoch']
        tmp_list.append([mic_pr, mic_rec, mic_f1,
                         mac_pr, mac_rec, mac_f1,
                         all_pr, all_rec, all_f1,
                         micro_tp, micro_n, micro_m,
                         macro_tp, macro_n, macro_m,
                         all_tp, all_n, all_m,
                         epoch])
    if len(tmp_list) > 0:
        tmp_list = np.stack(tmp_list, axis=0)
        new_df = pd.DataFrame(tmp_list, columns=_COL_NAME)
        epoch_file = os.path.join(
            opt['output_dir_name'], 'epoch_metrics.csv'
        )
        new_df.to_csv(epoch_file, index=False)

def choose_best_epoch(opt, criterion='all_f1'):
    assert criterion in ['micro_precision', 'micro_recall', 'micro_f1',
                 'macro_precision', 'macro_recall', 'macro_f1',
                 'all_precision',   'all_recall',   'all_f1'], f"Unsupported criterion:{criterion}!"
    epoch_file = os.path.join(
            opt['output_dir_name'], 'epoch_metrics.csv'
    )
    pick_columns = ['micro_tp', 'micro_n', 'micro_m', 'macro_tp', 'macro_n', 'macro_m',
                        'all_tp', 'all_n', 'all_m']
    
    best_res_file = os.path.join(
            opt['output_dir_name'], 'best_res.csv'
        )
    
    if os.path.exists(epoch_file):
        df = pd.read_csv(epoch_file)
        # sort in descending order, pick the one with highest performance in criterion.
        # get corresponding digits, extract them, save in another file.
        df.sort_values(criterion, inplace=True, ascending=False)
        df = df.reset_index(drop=True)
        df = df.loc[[0], pick_columns]
        
    else:
        micro_tp, micro_n, micro_m = [0] * 3
        macro_tp, macro_n, macro_m = [0] * 3
        all_tp, all_n, all_m = [0] * 3
        
        anno_df = pd.read_csv(opt['anno_csv'])
        tmp_df = anno_df[(anno_df['type_idx'] == 1 )
                     & (anno_df['subject'] == opt['subject'])]
        macro_m += len(tmp_df)
        
        tmp_df = anno_df[(anno_df['type_idx'] == 2 )
                     & (anno_df['subject'] == opt['subject'])]
        micro_m += len(tmp_df)
        
        all_m += micro_m + macro_m
        tmp_list = np.array([
                         micro_tp, micro_n, micro_m,
                         macro_tp, macro_n, macro_m,
                         all_tp, all_n, all_m]).reshape(1, 9)
        df = pd.DataFrame(tmp_list, columns=pick_columns)
    
    df.to_csv(best_res_file, index=False)