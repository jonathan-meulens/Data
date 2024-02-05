
import torch
import numpy as np



def get_ND_bounding_box(volume, margin = None):
    """
    get the bounding box of nonzero region in an ND volume
    """
    input_shape = volume.shape
    if(margin is None):
        margin = [0] * len(input_shape)
    elif (margin is not None):
        margin = [margin] * len(input_shape)

    assert(len(input_shape) == len(margin))
    if isinstance(volume, np.ndarray):
        indxes = np.nonzero(volume)
    elif isinstance(volume, torch.Tensor):
        indxes = torch.nonzero(volume)
    else:
        raise TypeError("Unsupported data type. Expected numpy ndarray or torch Tensor.")
    idx_min = []
    idx_max = []
    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i])
    return idx_min, idx_max # REMEMBER TO STORE THIS VALUE IN OUTPUT_VARAIBLE


def set_ND_volume_roi_with_bounding_box_range(volume, bb_min, bb_max):
    """
    set a subregion to an nd image.
    """
    dim = len(bb_min)
    if(dim == 2):
        out= volume[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1]))]
    elif(dim == 3):
        out = volume[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] +1),
                   range(bb_min[2], bb_max[2] +1))]
    elif(dim == 4):
        out= volume[np.ix_(range(bb_min[0], bb_max[0] + 1),
                   range(bb_min[1], bb_max[1] + 1),
                   range(bb_min[2], bb_max[2] + 1),
                   range(bb_min[3], bb_max[3] + 1))]
    else:
        raise ValueError("array dimension should be 2, 3 or 4")
    return out





