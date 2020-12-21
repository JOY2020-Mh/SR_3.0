import torch
import math
import torch.nn.functional as F
from ._CX_distance import CX_loss
import math
import random

# necessary now, but should eventually not be
import scipy.ndimage as ndi
import numpy as np

def transform_matrix_offset_center(matrix, x, y):
    """Apply offset to a transform matrix so that the image is
    transformed about the center of the image. 
    NOTE: This is a fairly simple operaion, so can easily be
    moved to full torch.
    Arguments
    ---------
    matrix : 3x3 matrix/array
    x : integer
        height dimension of image to be transformed
    y : integer
        width dimension of image to be transformed
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform, fill_mode='nearest', fill_value=0.):
    """Applies an affine transform to a 2D array, or to each channel of a 3D array.
    NOTE: this can and certainly should be moved to full torch operations.
    Arguments
    ---------
    x : np.ndarray
        array to transform. NOTE: array should be ordered CHW
    
    transform : 3x3 affine transform matrix
        matrix to apply
    """
    x = x.astype('float32')
    transform = transform_matrix_offset_center(transform, x.shape[1], x.shape[2])
    final_affine_matrix = transform[:2, :2]
    final_offset = transform[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
            final_offset, order=0, mode=fill_mode, cval=fill_value) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    return x


class Translation(object):

    def __init__(self, 
                 translation_range, 
                 fill_mode='constant',
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly translate an image some fraction of total height and/or
        some fraction of total width. If the image has multiple channels,
        the same translation will be applied to each channel.
        Arguments
        ---------
        translation_range : two floats between [0, 1) 
            first value:
                fractional bounds of total height to shift image
                image will be horizontally shifted between 
                (-height_range * height_dimension, height_range * height_dimension)
            second value:
                fractional bounds of total width to shift image 
                Image will be vertically shifted between 
                (-width_range * width_dimension, width_range * width_dimension)
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
        fill_value : float
            the value to fill the empty space with if fill_mode='constant'
        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        if isinstance(translation_range, float):
            translation_range = (translation_range, translation_range)
        self.height_range = translation_range[0]
        self.width_range = translation_range[1]
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        # height shift
        # if self.height_range > 0:
        tx = self.height_range * x.size(1)
        # else:
        #     tx = 0
        # # width shift
        # if self.width_range > 0:
        ty = self.width_range * x.size(2)
        # else:
        #     ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.lazy:
            return translation_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), 
                translation_matrix, fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), translation_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed


class Affine(object):

    def __init__(self, 
                 translation_range=None,
                 fill_mode='constant',
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0.):
        """Perform an affine transforms with various sub-transforms, using
        only one interpolation and without having to instantiate each
        sub-transform individually.
        Arguments
        ---------
        rotation_range : one integer or float
            image will be rotated between (-degrees, degrees) degrees
        translation_range : a float or a tuple/list w/ 2 floats between [0, 1)
            first value:
                image will be horizontally shifted between 
                (-height_range * height_dimension, height_range * height_dimension)
            second value:
                Image will be vertically shifted between 
                (-width_range * width_dimension, width_range * width_dimension)
        shear_range : float
            radian bounds on the shear transform
        zoom_range : list/tuple with two floats between [0, infinity).
            first float should be less than the second
            lower and upper bounds on percent zoom. 
            Anything less than 1.0 will zoom in on the image, 
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in, 
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
            ProTip : use 'nearest' for discrete images (e.g. segmentations)
                    and use 'constant' for continuous images
        fill_value : float
            the value to fill the empty space with if fill_mode='constant'
        target_fill_mode : same as fill_mode, but for target image
        target_fill_value : same as fill_value, but for target image
        """
        self.transforms = []

        if translation_range:
            translation_tform = Translation(translation_range, lazy=True)
            self.transforms.append(translation_tform)

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value

    def __call__(self, x, y=None):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0](x)
        for tform in self.transforms[1:]:
            tform_matrix = np.dot(tform_matrix, tform(x)) 

        x = torch.from_numpy(apply_transform(x.numpy(), tform_matrix,
            fill_mode=self.fill_mode, fill_value=self.fill_value))

        if y:
            y = torch.from_numpy(apply_transform(y.numpy(), tform_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
            return x, y
        else:
            return x

def extract_image_patches(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
    
    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0,4,5,1,2,3).contiguous()
    
    return patches.view(b,-1,patches.shape[-2], patches.shape[-1])

def crop_quarters(feature_tensor):
    N, fC, fH, fW = feature_tensor.shape
    quarters_list = []
    quarter_size = [N, fC, round(fH / 2), round(fW / 2)]
    quarters_list.append(feature_tensor[0:N,0:fC,0:round(fH / 2),0:round(fW / 2)])
    quarters_list.append(feature_tensor[0:N,0:fC,round(fH / 2):round(fH / 2) + round(fH / 2),0:round(fW / 2)])
    quarters_list.append(feature_tensor[0:N,0:fC,0:round(fH / 2),round(fW / 2):round(fW / 2)+round(fW / 2)])
    quarters_list.append(feature_tensor[0:N,0:fC,round(fH / 2):round(fH / 2) + round(fH / 2),round(fW / 2):round(fW / 2)+round(fW / 2)])
    feature_tensor = torch.concat(quarters_list, axis=0)
    return feature_tensor

def random_sampling(tensor_NCHW, n, indices=None):
    N, C, H, W  = tensor_NCHW.shape
    S = H * W
    # NCHW to NHWC
    tensor_NHWC = tensor_NCHW.permute(0, 2, 3, 1)
    tensor_NSC = torch.reshape(tensor_NHWC, [N, S, C])
    # all_indices = list(range(S))
    shuffled_indices = torch.randperm(S)
    indices = torch.gather(shuffled_indices, 0, torch.arange(0,n)) if indices is None else indices
    res = torch.index_select(tensor_NSC, 1, indices.cuda())
    # res: NSC -> NCS
    return res.permute(0, 2, 1), indices


def random_pooling(feats, output_1d_size=100):
    is_input_tensor = type(feats) is torch.Tensor
    # convert a single tensor to list
    if is_input_tensor:
        feats = [feats]

    N, C, H, W = feats[0].shape
    feats_sampled_0, indices = random_sampling(feats[0], output_1d_size ** 2)
    res = [feats_sampled_0]
    for i in range(1, len(feats)):
        feats_sampled_i, _ = random_sampling(feats[i], -1, indices)
        res.append(feats_sampled_i)
    # res：NxCxS
    res = [torch.reshape(feats_sampled_i, [N, C, output_1d_size, output_1d_size]) for feats_sampled_i in res]
    if is_input_tensor:
        return res[0]
    return res

def CX_loss_helper(vgg_A, vgg_B, CX_config):
    # vgg_A: NxCxHxW vgg_B: NxCxHxW
    if CX_config.crop_quarters is True:
        vgg_A = crop_quarters(vgg_A)
        vgg_B = crop_quarters(vgg_B)

    N, fC, fH, fW= vgg_A.shape
    if fH * fW <= CX_config.max_sampling_1d_size ** 2:
        # print(' #### Skipping pooling for CX....')
        pass
    else:
        # print(' #### pooling for CX %d**2 out of %dx%d' % (CX_config.max_sampling_1d_size, fH, fW))
        vgg_A, vgg_B = random_pooling([vgg_A, vgg_B], output_1d_size=CX_config.max_sampling_1d_size)
    # vgg_A: NxCxHxW vgg_B: NxCxHxW
    cx_loss,_ = CX_loss(vgg_A, vgg_B,
        distance=CX_config.Dist,
        nnsigma=CX_config.nn_stretch_sigma,
        w_spatial=CX_config.w_spatial)
    return cx_loss



if __name__ == "__main__":
    # batch_size = 128
    # channels = 16
    # height, width = 32, 32
    # x = torch.randn(batch_size, channels, height, width)
    # print(x.shape)
    # patches = extract_image_patches(x, kernel=3)
    # print(patches.shape)
    T_features = torch.randn((9,224,224,64))
    N,H,W,C=T_features.shape
    # res = torch.gather(tensor_NSC, dim=1, index=indices)
    # random_sampling(T_features)
    # rows = torch.tensor(np.arange(H)/(H) * 255.).float()
    # cols = torch.tensor(np.arange(W)/(W) * 255.).float()
    # features_grid = torch.meshgrid(rows, cols)[::-1]
    # features_grid = torch.cat([torch.unsqueeze(features_grid_i, 2) for features_grid_i in features_grid], axis=2)
    # features_grid = torch.unsqueeze(features_grid, 0)
    # features_grid = features_grid.repeat(N, 1, 1, 1)
    # print(features_grid.shape)
    S = H * W
    tensor_NSC = torch.reshape(T_features, [N, S, C])
    # all_indices = list(range(S))
    shuffled_indices = torch.randperm(S)
    n = 63 ** 2
    indices = torch.gather(shuffled_indices, 0, torch.arange(0,n))
    # indices = torch.unsqueeze(torch.unsqueeze(indices, dim=0), dim=2)
    print(tensor_NSC.shape)
    res = torch.index_select(tensor_NSC, 1, indices)
    print(res.shape)
    
    # print(type(T_features) is torch.Tensor)