import os

from keras.preprocessing.image import ImageDataGenerator, array_to_img
from PIL import Image
import numpy as np

class DataFeeder(object):
    def __init__(self, load_img_dir, load_mask_dir, batch_size=64, size=(64, 64),mask_ratio = 0.5):
        self.load_img_dir = load_img_dir
        self.load_mask_dir = load_mask_dir
        self.batch_size = batch_size
        self.size = size
        self.generator = ImageDataGenerator().flow_from_directory(self.load_img_dir, target_size=size, batch_size = batch_size)
        self.mask_ratio = mask_ratio
        self.mask_generator = ImageDataGenerator().flow_from_directory(self.load_mask_dir, target_size =(int(size[0]*self.mask_ratio),int(size[1]*self.mask_ratio))\
                                                 , batch_size = batch_size,color_mode='grayscale')

    def mask_randomly_expand(self, imgs, masks):
        img_height = imgs.shape[1]
        mask_height = masks.shape[1]
        mask_width = mask_height
        y1 = np.random.randint(0, img_height - mask_height, imgs.shape[0])
        y2 = y1 + mask_height
        x1 = np.random.randint(0, img_height - mask_width, imgs.shape[0])
        x2 = x1 + mask_width

        expanded_masks = np.empty((self.batch_size, self.size[0], self.size[1], 1))

        for i, mask in enumerate(masks):
            expanded_mask = np.zeros((self.size[0], self.size[1], 1))
            _y1, _y2, _x1, _x2 = y1[i], y2[i], x1[i], x2[i]
            expanded_mask[_y1:_y2, _x1:_x2, :] = mask
            expanded_masks[i] = expanded_mask
        return expanded_masks
    
    
    def fetch_data(self):
        imgs, _ = next(self.generator)
        masks, _ = next(self.mask_generator)


        if imgs.shape[0] == self.batch_size and masks.shape[0] == self.batch_size:
            masks = self.mask_randomly_expand(imgs, masks)/255.
            masked_imgs = imgs*(1 - masks)
            return (imgs/255., masks, masked_imgs/255.)
        else:
            return self.fetch_data()


    
    def save_images(self, arrays, names, concat, save_dir='save',sketch = 5):
        if not isinstance(names, list):
            names = [names]
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        assert sketch<arrays.shape[0]
        
        arrays = arrays[:sketch,:,:,:]
        if not concat:
            for array, name in zip(arrays, names):
                image = array_to_img(array).resize((120, 120))
                image.save("{}.png".format(name), quality=100)
        else:
            canvas = Image.new('RGB', (120*len(arrays), 120), (255, 255, 255))
            for i, array in enumerate(arrays):
                image = array_to_img(array).resize((120, 120))
                canvas.paste(image, (i*120, 0))
            canvas.save(os.path.join(save_dir, "{}.png".format(names[0])), quality=100)

if __name__ == '__main__':
    data_feeder = DataFeeder('./dir_test', batch_size=2)
    data = data_feeder.fetch_data()
