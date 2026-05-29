# uncompyle6 version 3.9.0
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.7.11 (default, Jul 27 2021, 14:32:16) 
# [GCC 7.5.0]
# Embedded file name: /userhome/BBBBBBBBB/PQA-Net-mindspore/VideoTransforms.py
# Compiled at: 2022-12-14 10:37:02
# Size of source mod 2**32: 10040 bytes
import numpy as np, mindspore as ms
import mindspore.nn as nn
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

class ClipWrapper(object):
    __doc__ = 'A wrapper to accommodate a clip in the form of a list of PIL images.\n    Only valid for deterministic spatial transform\n    Args:\n        transform: a deterministic spatial transform for a single PIL image\n    '

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, imgs):
        """
        transform a list of images
        :param imgs: a list of PIL images.
        :return: None
        """
        assert isinstance(imgs, list)
        return [self.transform(img) for img in imgs]


class RandomTemporalClip(object):
    __doc__ = 'Take random clips of the video.'

    def __init__(self, clip_num=1, clip_size=None, stride=1):
        if not isinstance(clip_num, int):
            raise AssertionError
        else:
            assert clip_num > 0
            if clip_size is not None:
                assert isinstance(clip_size, int)
                if not clip_size > 0:
                    raise AssertionError
        assert isinstance(stride, int)
        assert stride > 0
        self.clip_num = clip_num
        self.clip_size = clip_size
        self.stride = stride

    def __call__(self, frame_indices):
        """
        :param frame_indices: sorted list of frame names or indices
        :return:
        """
        if self.clip_size is None:
            start_idxs = np.random.permutation(np.arange(0, len(frame_indices), self.stride))
            start_idxs = start_idxs[:self.clip_num]
        else:
            start_idxs = np.random.permutation(np.arange(0, len(frame_indices) - self.clip_size + 1, self.stride))
            start_idxs = start_idxs[:self.clip_num]
        if self.clip_size is None:
            frame_indices = [frame_indices[iii] for iii in start_idxs]
        else:
            frame_indices = [frame_indices[iii:iii + self.clip_size] for iii in start_idxs]
        return frame_indices


class RandomCropWithStrides(object):
    __doc__ = 'Crop the given PIL.image at random location of a fixed grid\n\n    Args:\n        size (sequence or int): Desired output size of the crop. If size is an\n            int instead of sequence like (h, w), a square crop (size, size) is\n            made.\n        stride (sequence or int): the stride of image cropping\n    '

    def __init__(self, size, stride, is_clip=False):
        if not isinstance(size, (int, tuple)):
            raise AssertionError
        else:
            if isinstance(size, int):
                self.size = (
                 size, size)
            else:
                assert len(size) == 2
                self.size = size
            assert isinstance(stride, (int, tuple))
            if isinstance(stride, int):
                self.stride = (
                 stride, stride)
            else:
                assert len(stride) == 2
            self.stride = stride
        assert isinstance(is_clip, bool)
        self.is_clip = is_clip

    def __call__(self, target):
        """
        :param target: a PIL.Image or a clip in the form of a list of images to be cropped
        :return: Cropped PIL.Image or clip
        """
        if isinstance(target, list):
            assert len(target) > 0
            return self._crop_clip_(target)
        return self._crop_img_(target)

    def _crop_img_(self, img):
        """
        :param img: PIL.Image to be cropped
        :return: Cropped PIL.Image
        """
        w, h = img.size
        h_cropped, w_cropped = self.size
        h_stride, w_stride = self.stride
        if w == w_cropped:
            if h == h_cropped:
                return img
        w_max_idx = (w - w_cropped) // w_stride
        h_max_idx = (h - h_cropped) // h_stride
        w1 = np.random.randint(low=0, high=(w_max_idx + 1)) * w_stride
        h1 = np.random.randint(low=0, high=(h_max_idx + 1)) * h_stride
        return img.crop((w1, h1, w1 + w_cropped, h1 + h_cropped))

    def _crop_clip_(self, clip):
        """
        :param clip: a list of PIL.Image of the same size to be cropped
        :return: a list of cropped PIL.Image
        """
        w, h = clip[0].size
        h_cropped, w_cropped = self.size
        h_stride, w_stride = self.stride
        if w == w_cropped:
            if h == h_cropped:
                return clip
        w_max_idx = (w - w_cropped) // w_stride
        h_max_idx = (h - h_cropped) // h_stride
        w1 = np.random.randint(low=0, high=(w_max_idx + 1)) * w_stride
        h1 = np.random.randint(low=0, high=(h_max_idx + 1)) * h_stride
        return [img.crop((w1, h1, w1 + w_cropped, h1 + h_cropped)) for img in clip]


class DenseTemporalClip(object):
    __doc__ = 'Transform a video segment into clips.'

    def __init__(self, clip_size=None):
        if clip_size is not None:
            assert isinstance(clip_size, int)
            assert clip_size >= 1
        self.clip_size = clip_size

    def __call__(self, frame_indices):
        """
        Transform frame_indices into a list of non-overlapping clips (or frame_indices itself if self.clip_size is None)
        :param frame_indices: list of frame indices
        :return: clips: clips = frame_indices if clip_size is None, or a list of clips
        """
        if self.clip_size is None:
            clips = frame_indices
        else:
            if self.clip_size > len(frame_indices):
                raise ValueError('The length of input video segment is too short.')
            start_idxs = range(0, len(frame_indices) - self.clip_size + 1, self.clip_size)
            clips = [frame_indices[start_idx:start_idx + self.clip_size] for start_idx in start_idxs]
        return clips


class DenseSpatialCrop(object):
    __doc__ = 'Densely crop an image, where stride is equal to the output size.\n\n    Args:\n        output_size (tuple or int): Desired output size. If int, square crop\n            is made.\n    '

    def __init__(self, output_size, stride):
        if not isinstance(output_size, (int, tuple)):
            raise AssertionError
        else:
            if isinstance(output_size, int):
                self.output_size = (
                 output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size
            assert isinstance(stride, (int, tuple))
            if isinstance(stride, int):
                self.stride = (
                 stride, stride)
            else:
                assert len(stride) == 2
            self.stride = stride

    def __call__(self, image):
        w, h = image.size[:2]
        new_h, new_w = self.output_size
        stride_h, stride_w = self.stride
        h_start = np.arange(0, h - new_h, stride_h)
        w_start = np.arange(0, w - new_w, stride_w)
        patches = [image.crop((wv_s, hv_s, wv_s + new_w, hv_s + new_h)) for hv_s in h_start for wv_s in iter(w_start)]
        to_tensor = vision.ToTensor()
        patches = [to_tensor(patch) for patch in patches]
        patches = ops.stack(patches, axis=0)
        return patches


class DenseSpatialCrop_collate(object):
    __doc__ = 'Densely crop an image, where stride is equal to the output size.\n\n    Args:\n        output_size (tuple or int): Desired output size. If int, square crop\n            is made.\n    '

    def __init__(self, output_size, stride):
        if not isinstance(output_size, (int, tuple)):
            raise AssertionError
        else:
            if isinstance(output_size, int):
                self.output_size = (
                 output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size
            assert isinstance(stride, (int, tuple))
            if isinstance(stride, int):
                self.stride = (
                 stride, stride)
            else:
                assert len(stride) == 2
            self.stride = stride

    def __call__(self, image):
        w, h = image.size[:2]
        new_h, new_w = self.output_size
        stride_h, stride_w = self.stride
        h_start = np.arange(0, h - new_h, stride_h)
        w_start = np.arange(0, w - new_w, stride_w)
        patches = [image.crop((wv_s, hv_s, wv_s + new_w, hv_s + new_h)) for hv_s in h_start for wv_s in iter(w_start)]
        to_tensor = vision.ToTensor()
        patches = [to_tensor(patch) for patch in patches]
        patches = ops.stack(patches, axis=0)
        return patches


class RandomHorizontalFlipClip(object):
    __doc__ = 'Horizontally flip the given clip in the form of a list of PIL Images randomly with a given probability.\n    Args:\n        p (float): probability of the image being flipped. Default value is 0.5\n    '

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (a list of PIL Image): Clip to be flipped.
        Returns:
            a list of PIL Image: Randomly flipped clip.
        """
        if np.random.rand() < self.p:
            return [vision.HorizontalFlip()(img) for img in clip]
        return clip

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomPatchSelection(object):
    __doc__ = '\n    Randomly select a patch index\n    Args:\n        lo (positive int): lowest possible patch index\n        hi (positive int): highest possible patch index\n\n    '

    def __init__(self, lo, hi):
        assert lo > 0
        assert hi >= lo
        self.lo = int(lo)
        self.hi = int(hi)

    def __call__(self, clips):
        clip_num = len(clips)
        patch_indices = np.random.randint((self.lo), (self.hi + 1), size=clip_num)
        frame_indices = [(clips[iii], patch_indices[iii]) for iii in range(clip_num)]
        return frame_indices