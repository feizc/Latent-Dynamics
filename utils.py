import torch 
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone 
from torchvision.models.detection.rpn import AnchorGenerator


SPECIAL_TOKENS = ["[bos]", "[eos]", "[dyn]", "[img]", "[txt]"] 
SPECIAL_TOKENS_DICT = {'bos_token': "[bos]", 'eos_token': "[eos]", 'additional_special_tokens': ["[dyn]", "[img]", "[txt]"],}


def accuracy_compute(logits, labels, top_k=5, ignore_index=-100): 
    bsz, seq_len, _ = logits.size()
    logits = logits.contiguous().view(bsz*seq_len, -1)
    _, idx = torch.topk(logits, top_k, -1) 
    correct = idx.eq(labels.view(-1, 1).expand_as(idx)) 
    correct_total = correct.view(-1).float().sum().item()
    nums = labels.view(-1).detach().cpu().numpy()
    length = 0 
    for num in nums:
        if num != ignore_index:
            length += 1
    return correct_total / float(length) 


def fasterrcnn_resnet152_fpn(pretrained=False, progress=True,
                            num_classes=91, pretrained_backbone=False,
                             trainable_backbone_layers=3, **kwargs):
    assert trainable_backbone_layers <= 5 and trainable_backbone_layers >= 0
    # dont freeze any layers if pretrained model or backbone is not used
    if not (pretrained or pretrained_backbone):
        trainable_backbone_layers = 5
    backbone = resnet_fpn_backbone('resnet152', pretrained_backbone) 
    model = FasterRCNN(backbone, num_classes, **kwargs)
    return model 


def faster_rcnn_initilize(model_path): 
    model = fasterrcnn_resnet152_fpn(pretrained=False) 
    model.load_state_dict(torch.load(model_path)) 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 30)
    return model 
     

# extract faster-rcnn features of 2048 dimension 
def faster_rcnn_feature_extraction(model, image): 
    # hook for fc6 layer 
    model.eval()
    outputs = []
    hook = model.backbone.register_forward_hook(
        lambda self, input, output: outputs.append(output)) 

    res = model(image)
    hook.remove() 

    selected_rois = model.roi_heads.box_roi_pool(
        outputs[0], [r['boxes'] for r in res], [i.shape[-2:] for i in image])
    selected_rois = model.roi_heads.box_head.fc6(selected_rois.view(-1, 12544))

    return selected_rois 
    