
def add_seed(cfg):
    if cfg.DATASETS.dataname == 'pascal':
        if cfg.DATASETS.SPLIT == 0:
            seed = 4604572
        elif  cfg.DATASETS.SPLIT == 1:
            seed = 7743485
        elif  cfg.DATASETS.SPLIT == 2:
            seed = 5448843
        elif  cfg.DATASETS.SPLIT == 3:
            seed = 2534673

    if cfg.DATASETS.dataname == 'coco':
        if cfg.DATASETS.SPLIT == 0:
            seed = 8420323
        elif  cfg.DATASETS.SPLIT == 1:
            seed = 27163933
        elif  cfg.DATASETS.SPLIT == 2:
            seed = 8162312
        elif  cfg.DATASETS.SPLIT == 3:
            seed = 3391510
    if cfg.DATASETS.dataname == 'c2pv':
        seed = 321
    return ['SEED', seed]