import numpy as np
from PIL import Image
import tensorflow as tf
import time

def make_dataset(root_txt_path, labels):
    with open(root_txt_path, "r") as f_txt:
        image_list = f_txt.readlines()  # 读取全部内容 ，并以列表方式返回
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        images = []
        for val in image_list:
            if len(val)>5:
                if len(val.split()) > 2:
                    images.append((' '.join(val.split()[:-1]), float(val.split()[-1])))
                else:
                    images.append((val.split()[0], float(val.split()[1])))
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList():
    def __init__(self, image_list, batch_size, pic_resize, mode='RGB'):
        self.imgs = make_dataset(image_list, labels=None)
        self.pic_resize = pic_resize
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.dataset = tf.data.Dataset.from_tensor_slices(np.arange(self.length()))
        self.dataset = self.dataset.shuffle(buffer_size=self.length(), seed=int(time.time()%1e5), reshuffle_each_iteration=True) #每个epoch的seed必须不一样，不然等于没shuffle
        self.dataset = self.dataset.map(lambda x: tf.numpy_function(self.getitem, [x], [tf.float32, tf.float32, tf.string]), 
            num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
        self.dataset = self.dataset.repeat(1) #每个epoch重新创建dataset
        self.dataset = self.dataset.batch(batch_size, drop_remainder=False)
        self.dataset = self.dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) #要放在batch之后

    def getitem(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        read_frame = np.array(img, dtype=np.float32) #hwc
        # print(read_frame.shape)
        read_frame = tf.convert_to_tensor(read_frame)
        read_frame = tf.image.resize(read_frame, [self.pic_resize,self.pic_resize], method=tf.image.ResizeMethod.BILINEAR)
        read_frame = read_frame / 255.0
        read_frame = (read_frame - [0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5]
        img = tf.transpose(read_frame, perm=[2,0,1]) #chw

        return img, np.float32(target), path #target是一个float数

    def length(self):
        return len(self.imgs)

if __name__ == '__main__':
    img = rgb_loader('/userhome/IT-PCQA/scripts/SJTU-PCQA/projection/projection_splicing/ULB Unicorn_9.png')
    img = np.array(img, dtype=np.float32)
    print(img.shape)