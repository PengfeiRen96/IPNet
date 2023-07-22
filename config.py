JOINT = {
    'nyu':14,
    'dexycb':21,
    'ho3d':21
}

STEP = {
    'nyu': 25,
    'dexycb':10,
    'ho3d':25
}

EPOCH = {
    'nyu': 30,
    'dexycb':15,
    'ho3d':30
}

CUBE = {
    'nyu': [250, 250, 250],
    'dexycb':[250, 250, 250],
    'ho3d': [280, 280, 280],
}


class Config(object):
    phase = 'train'
    root_dir = '' # set your path

    dataset = 'dexycb'# ['nyu', 'dexycb', 'ho3d']
    ho3d_version = 'v3'
    dexycb_setup = 's0' # ['s0', 's1', 's2', 's3]

    model_save = ''
    add_info = ''
    save_dir = './'
    net = 'PointNet' # ['IPNet-convnext-tiny', 'IPNet_MANO-convnext-tiny', 'PointMLP', 'PointNet', 'mix-Point2', 'mix-PointMLP']

    load_model = ''
    finetune_dir = ''

    gpu_id = 0
    joint_num = JOINT[dataset]

    batch_size = 32
    input_size = 128
    cube_size = CUBE[dataset]
    center_type = 'refine'  # ['joint_mean', 'refine']
    loss_type = 'L1Loss'
    augment_para = [10, 0.2, 180]

    lr = 0.001
    max_epoch = EPOCH[dataset]
    step_size = STEP[dataset]
    opt = 'adamw'
    scheduler = 'step'
    downsample = 2

    awr = True
    coord_weight = 100
    deconv_weight = 1

    feature_type = ['weight_offset']
    feature_para = [0.8]
    # stage_type = [1]  # Pixel-wise Regression (ResNet ConvNeXt)
    stage_type = [3]  # Point-wise Regression (PointMLP PointNet)
    # stage_type = [1, 3] # 1-stage IPNet
    # stage_type = [1, 3,  3,  3] # 3-stage IPNet
    # stage_type = [1, 3, 5, 3, 5, 3, 5] # 3-stage IPNet_MANO


opt = Config()

