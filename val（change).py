import time
import torch
import numpy as np
from modules import utils
import modules.evaluation as evaluate
from modules.box_utils import decode
from modules.utils import get_individual_labels
import torch.utils.data as data_utils
from data import custum_collate
from torchvision.ops import nms  # 引入GPU加速的NMS

logger = utils.get_logger(__name__)


def val(args, net, val_dataset):
    val_data_loader = data_utils.DataLoader(val_dataset, args.BATCH_SIZE, num_workers=args.NUM_WORKERS,
                                            shuffle=False, pin_memory=True, collate_fn=custum_collate)
    args.MODEL_PATH = args.SAVE_ROOT + 'model_{:06d}.pth'.format(args.EVAL_EPOCHS[0])
    logger.info('Loaded model from :: ' + args.MODEL_PATH)
    net.load_state_dict(torch.load(args.MODEL_PATH))

    # 提前准备好计算MIOU等评估指标所需的数据结构
    label_types = args.label_types + ['ego_action']
    all_classes = args.all_classes + [args.ego_classes]

    mAP, ap_all, ap_strs = validate(args, net, val_data_loader, val_dataset, args.EVAL_EPOCHS[0], label_types,
                                    all_classes)
    # 输出各类指标信息
    for nlt in range(args.num_label_types + 1):
        for ap_str in ap_strs[nlt]:
            logger.info(ap_str)
        ptr_str = '\n{:s} MEANAP:::=> {:0.5f}'.format(label_types[nlt], mAP[nlt])
        logger.info(ptr_str)


def validate(args, net, val_data_loader, val_dataset, iteration_num, label_types, all_classes):
    """Test a FPN network on an image database."""

    iou_thresh = args.IOU_THRESH
    num_samples = len(val_dataset)
    logger.info('Validating at ' + str(iteration_num) + ' number of samples:: ' + str(num_samples))

    # 计算和日志打印时间的优化
    print_time = True
    val_step = 20
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    activation = torch.nn.Sigmoid().cuda()

    ego_pds = []
    ego_gts = []

    det_boxes = []
    gt_boxes_all = []

    for nlt in range(args.num_label_types):
        numc = args.num_classes_list[nlt]
        det_boxes.append([[] for _ in range(numc)])
        gt_boxes_all.append([])

    net.eval()
    with torch.no_grad():
        for val_itr, (images, gt_boxes, gt_targets, ego_labels, batch_counts, img_indexs, wh) in enumerate(
                val_data_loader):

            # 使用GPU时尽量避免过多同步
            batch_size = images.size(0)

            images = images.cuda(0, non_blocking=True)
            decoded_boxes, confidence, ego_preds = net(images)
            ego_preds = activation(ego_preds).cpu().numpy()
            ego_labels = ego_labels.numpy()
            confidence = activation(confidence)

            if print_time and val_itr % val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                logger.info('Forward Time {:0.3f}'.format(tf - ts))

            seq_len = gt_targets.size(1)
            for b in range(batch_size):

                for s in range(seq_len):
                    if args.DATASET == 'ava' and batch_counts[b, s] < 1:
                        continue

                    if ego_labels[b, s] > -1:
                        ego_pds.append(ego_preds[b, s, :])
                        ego_gts.append(ego_labels[b, s])

                    width, height = wh[b][0], wh[b][1]
                    gt_boxes_batch = gt_boxes[b, s, :batch_counts[b, s], :].numpy()
                    gt_labels_batch = gt_targets[b, s, :batch_counts[b, s]].numpy()

                    decoded_boxes_frame = decoded_boxes[b, s].clone().cuda()  # 将框数据移动到GPU

                    cc = 0
                    for nlt in range(args.num_label_types):
                        num_c = args.num_classes_list[nlt]
                        tgt_labels = gt_labels_batch[:, cc:cc + num_c]
                        frame_gt = get_individual_labels(gt_boxes_batch, tgt_labels)
                        gt_boxes_all[nlt].append(frame_gt)

                        for cl_ind in range(num_c):
                            scores = confidence[b, s, :, cc].clone().squeeze().cuda()  # 将confidence移到GPU
                            cc += 1
                            # 使用GPU加速的NMS
                            keep = nms(decoded_boxes_frame, scores, iou_threshold=0.2)  # 使用 NMS
                            cls_dets = decoded_boxes_frame[keep]
                            det_boxes[nlt][cl_ind].append(cls_dets)
                count += 1

            if print_time and val_itr % val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                logger.info('detections done: {:d}/{:d} time taken {:0.3f}'.format(count, num_samples, te - ts))
                torch.cuda.synchronize()
                ts = time.perf_counter()

    # 在循环外集中进行一次同步
    torch.cuda.synchronize()

    logger.info('Evaluating detections for epoch number ' + str(iteration_num))
    mAP, ap_all, ap_strs = evaluate.evaluate(gt_boxes_all, det_boxes, all_classes, iou_thresh=iou_thresh)
    mAP_ego, ap_all_ego, ap_strs_ego = evaluate.evaluate_ego(np.asarray(ego_gts), np.asarray(ego_pds), args.ego_classes)

    # 返回所有的mAP、AP信息
    return mAP + [mAP_ego], ap_all + [ap_all_ego], ap_strs + [ap_strs_ego]

