# Importing Libraries
import torchreid
from ELEDataset import NewDataset

import torch
import torch.nn as nn

torchreid.data.register_image_dataset('embodied-yt-2', NewDataset)

datamanager = torchreid.data.ImageDataManager(
    root='../',
    sources='embodied-yt-2',
    targets='embodied-yt-2',
    height=256,
    width=128,
    batch_size_train=256,
    batch_size_test=256,
    transforms=['random_flip', 'random_erase']
)

model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True,
    use_gpu=True
)

torchreid.utils.load_pretrained_weights(model=model, weight_path=r'./osnet_ain_x1_0_imagenet.pth')

model = nn.DataParallel(model).cuda()


optimizer = torchreid.optim.build_optimizer(
    model,
    optim='amsgrad',
    lr=0.0015
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=60,
    gamma=0.1
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    use_gpu=True,
    label_smooth=True
)

start_epoch = torchreid.utils.resume_from_checkpoint(
    './log/run2/model/model.pth.tar-10',
    model=model,
    optimizer=optimizer,
    scheduler=scheduler
)

engine.run(
    save_dir='./log/run2/',
    max_epoch=150,
    eval_freq=10,
    print_freq=10,
    test_only=False,
    fixbase_epoch=10,
    start_epoch=start_epoch,
    open_layers=['classifier']
)
