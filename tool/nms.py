from pcdet.models.model_utils import model_nms_utils
def nms(cls_preds, box_preds, post_process_cfg):
    selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
    return selected