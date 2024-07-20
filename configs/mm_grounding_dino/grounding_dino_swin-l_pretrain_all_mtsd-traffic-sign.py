# 测试命令：CUDA_VISIBLE_DEVICES=2 python tools/test.py configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all_mtsd-traffic-sign.py weights_MMGD/grounding_dino_swin-l_pretrain_all-56d69e78.pth --work-dir work_dir_merged-test-res-nofinetune
# 微调命令：CUDA_VISIBLE_DEVICES=1,2,3,4 ./tools/dist_train.sh configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all_mtsd-traffic-sign.py 4 --work-dir work_dir_MMGD78_finetune_merged

_base_ = 'grounding_dino_swin-t_pretrain_obj365.py'

# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth'  # noqa
load_from = '/mmdetection/weights_MMGD/grounding_dino_swin-l_pretrain_all-56d69e78.pth'

# data_root = '/mmdetection/MTSD_dataset/OpenDataLab___Mapillary_Traffic_Sign_Dataset/raw/'
# class_name = ('traffic sign', )
# data_root = '/mmdetection/BSTLD_dataset/dataset_test_rgb/'
# class_name = ('traffic light', )
data_root = '/mmdetection/images_train-test_bstld-mtsd/'
class_name = ('traffic light', "traffic sign", )

num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(220, 20, 60)])

num_levels = 5
model = dict(
    use_autocast=True,
    num_feature_levels=num_levels,
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=None),
    neck=dict(in_channels=[192, 384, 768, 1536], num_outs=num_levels),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))),
    bbox_head=dict(num_classes=num_classes))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        return_classes=True,
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file=data_root + 'merged_train_dataset.json',
        data_prefix=dict(img='train/')))

val_dataloader = dict(sampler=dict(type='DefaultSampler', shuffle=False), batch_size = 1, num_workers=4,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='merged_test_dataset.json',
        data_prefix=dict(img='test/')))

test_dataloader = val_dataloader

val_evaluator = dict(metric=['bbox', 'proposal'], ann_file=data_root + 'merged_test_dataset.json')

test_evaluator = val_evaluator

optim_wrapper = dict( 
    optimizer=dict(lr=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.0),
            'language_model': dict(lr_mult=0.0)
        }))

max_iter = 250000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=13000)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[210000],
        gamma=0.1)
]

max_epoch = 20
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=2)

