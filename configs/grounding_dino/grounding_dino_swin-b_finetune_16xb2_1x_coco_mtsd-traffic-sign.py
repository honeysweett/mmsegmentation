# 单GPU测试命令：CUDA_VISIBLE_DEVICES=6 python tools/test.py configs/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco_mtsd-traffic-sign.py weights_GD/grounding_dino_swin-b_finetune_16xb2_1x_coco_20230921_153201-f219e0c0.pth --work-dir xxxx

_base_ = [
    './grounding_dino_swin-t_finetune_16xb2_1x_coco.py',
]

# data_root = '/mmdetection/MTSD_dataset/OpenDataLab___Mapillary_Traffic_Sign_Dataset/raw/'
# class_name = ('traffic sign', )
data_root = '/mmdetection/BSTLD_dataset/dataset_test_rgb/'
class_name = ('traffic light', )
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

load_from = 'https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth'  # noqa
model = dict(
    type='GroundingDINO',
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.3,
        patch_norm=True),
    neck=dict(in_channels=[256, 512, 1024]),
    bbox_head=dict(num_classes=num_classes)
)

val_dataloader = dict(sampler=dict(type='DefaultSampler', shuffle=True), batch_size =4, num_workers=4,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='bstld_test_coco_format.json',
        data_prefix=dict(img='rgb/test/')))

# test_cfg=dict(score_thr=0.2) #不是这样定义的，会报参数未定义的错误！

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'bstld_test_coco_format.json')
test_evaluator = val_evaluator

optim_wrapper = dict( 
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),
            'language_model': dict(lr_mult=0.0)
        }))