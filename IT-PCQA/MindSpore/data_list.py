import numpy as np
from PIL import Image

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
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        self.imgs = make_dataset(image_list, labels)
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img[0], target, path #target是一个float数

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    img = rgb_loader('/userhome/IT-PCQA/scripts/SJTU-PCQA/projection/projection_splicing/ULB Unicorn_9.png')
    img = np.array(img, dtype=np.float32)
    print(img.shape)