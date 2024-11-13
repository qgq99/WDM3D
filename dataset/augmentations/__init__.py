import logging
import numpy as np 

from .augmentations import (
    RandomHorizontallyFlip,
    Compose,
)

# from config import cfg

aug_list = [RandomHorizontallyFlip]
                    
logger = logging.getLogger("monocd.augmentations")

def get_composed_augmentations(cfg):
    aug_params = cfg["aug_params"]
    augmentations = []
    for aug, aug_param in zip(aug_list, aug_params):
        if aug_param[0] > 0:
            augmentations.append(aug(*aug_param))
            logger.info("Using {} aug with params {}".format(aug, aug_param))

    return Compose(augmentations)
