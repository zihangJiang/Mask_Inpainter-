from keras import backend as K
import tensorflow as tf
import numpy as np

from keras.applications.vgg19 import VGG19
import os
class MaskInpainter(object):
    def __init__(self, net, data_feeder, sess, batch_size=64, size=128, dis_lr=1e-4, gen_lr=1e-4,load = False):
        self.net = net
        if load:
            net.generator.load_weights(os.path.join("save", "generator_{}.h5".format("Mi")),by_name = True)
        self.image_size = size
        self.data_feeder = data_feeder
        self.sess = sess
        self.batch_size = batch_size
        self.dis_lr = dis_lr
        self.gen_lr = tf.Variable(float(gen_lr),trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.gen_lr.assign(self.gen_lr * 0.9)
        self.built = False


    def gram_matrix(self,x):
        assert K.ndim(x) == 3
        if K.image_data_format() == 'channels_first':
            features = K.batch_flatten(x)#transfer tensor to 2-dim tensor(matrix), keep the 1st-dim unchange
        else:
            features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram

    def style_loss(self,style, combination):
        assert K.ndim(style) == 3
        assert K.ndim(combination) == 3
        S = self.gram_matrix(style)
        C = self.gram_matrix(combination)
        channels = 3
        #size = img_nrows * img_ncols
        size=self.image_size*self.image_size
        return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    def per_loss(self,style,combination):
        S=style
        C=combination
        return K.sum(K.abs(S-C))

    def new_loss(self,real,comp,fake):
        # combine the 3 images into a single Keras tensor
        input_tensor = K.concatenate([real,comp,fake], axis=0)

        # build the VGG19 network with our 3 images as input
        # the model will be loaded with pre-trained ImageNet weights
        model = VGG19(input_tensor=input_tensor,
                            weights='/data/anaconda/CVAEFaceShop/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
        print('Model loaded.')

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

        feature_layers = ['block1_conv1', 'block2_conv1',
                          'block3_conv1', 'block4_conv1',
                          'block5_conv1']
        sty_loss=0
        per_loss=0
        for layer_name in feature_layers:
            layer_features = outputs_dict[layer_name]
            real_features=layer_features[0,:,:,:]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_reference_features, real_features)
            sl+=style_loss(combination_features,real_features)
            pl = per_loss(style_reference_features, real_features)
            pl+=per_loss(combination_features,real_features)

            sty_loss += (1.0/ len(feature_layers)) * sl
            per_loss+=pl
        return sty_loss,per_loss


    def build(self):
        size = self.image_size
        
        # define inputs
        self.masked_image = tf.placeholder(tf.float32, shape=[None, size, size, 3])
        self.mask = tf.placeholder(tf.float32, shape=[None, size, size, 1])
        self.ones = tf.placeholder(tf.float32, shape=[None, size, size, 1])
        self.real_image = tf.placeholder(tf.float32, shape=[None, size, size ,3])
        self.fake_image , self.comp = self.net.generator([self.masked_image, self.mask, self.ones])
        
        # define loss
        self.loss_valid = K.sum(K.abs(((1-self.mask)*(self.fake_image - self.real_image))))
        self.loss_hole = K.sum(K.abs((self.mask)*(self.fake_image - self.real_image)))


        self.style_loss,self.per_loss=self.new_loss(self.real_image,self.comp,self.fake_image)

        #self.per_loss=per_loss(self.real_image,self.comp)
        #self.per_loss+=per_loss(self.real_image,self.fake_image)
        
        x = self.fake_image * self.mask
        a = K.square(x[:, :size - 1, :size - 1,:] - x[:, 1:, :size - 1,:])
        b = K.square(x[:, :size - 1, :size - 1, :] - x[:, :size - 1, 1:,:])
        self.loss_tv = K.sum(K.pow(a + b, 1.25))

        self.gen_loss = 6*self.loss_hole+self.loss_valid +0.1*self.loss_tv+0.05*self.per_loss+120*self.style_loss

        
        a = self.fake_image[:, :size - 1, :size - 1,:] * (K.abs(self.mask[:, :size - 1, :size - 1, :] - self.mask[:, 1:, :size - 1,:]) + K.abs(self.mask[:, :size - 1, :size - 1, :] - self.mask[:, :size - 1, 1:,:]))
        b = self.fake_image[:, 1:, :size - 1,:] * (K.abs(self.mask[:, :size - 1, :size - 1, :] - self.mask[:, 1:, :size - 1,:]) + K.abs(self.mask[:, :size - 1, :size - 1, :] - self.mask[:, :size - 1, 1:,:]))
        c = self.fake_image[:, :size - 1, 1:,:] * (K.abs(self.mask[:, :size - 1, :size - 1, :] - self.mask[:, 1:, :size - 1,:]) + K.abs(self.mask[:, :size - 1, :size - 1, :] - self.mask[:, :size - 1, 1:,:]))

        self.loss_boundary_fake = K.sum(K.abs(b-c))
        
        a = self.comp[:, :size - 1, :size - 1,:] * (K.abs(self.mask[:, :size - 1, :size - 1, :] - self.mask[:, 1:, :size - 1,:]) + K.abs(self.mask[:, :size - 1, :size - 1, :] - self.mask[:, :size - 1, 1:,:]))
        b = self.comp[:, 1:, :size - 1,:] * (K.abs(self.mask[:, :size - 1, :size - 1, :] - self.mask[:, 1:, :size - 1,:]) + K.abs(self.mask[:, :size - 1, :size - 1, :] - self.mask[:, :size - 1, 1:,:]))
        c = self.comp[:, :size - 1, 1:,:] * (K.abs(self.mask[:, :size - 1, :size - 1, :] - self.mask[:, 1:, :size - 1,:]) + K.abs(self.mask[:, :size - 1, :size - 1, :] - self.mask[:, :size - 1, 1:,:]))

        self.loss_boundary_comp = K.sum(K.abs(b-c))
        
        self.gen_loss = 6*self.loss_hole+self.loss_valid +0.1*self.loss_tv +10*self.loss_boundary_fake+30*self.loss_boundary_comp

        

        # initialize
        self.gen_updater = tf.train.AdamOptimizer(learning_rate=self.gen_lr,beta1=0., beta2=0.9).minimize(self.gen_loss, var_list=self.net.generator.trainable_weights)
        self.sess.run(tf.global_variables_initializer())
        self.built = True

    def train(self, epoch):
        if not self.built:
            self.build()
        for i in range(epoch):
            images, masks, masked_imgs = self.data_feeder.fetch_data()
            ones = np.ones_like(masks)
            feed_in = {self.masked_image: masked_imgs, self.mask: masks,
                    self.real_image: images, self.ones: ones}
            self.sess.run(self.gen_updater, feed_in)
            print("epoch: {}, gen_loss: {}, loss_hole: {}".format(i, self.sess.run(self.gen_loss, feed_in), self.sess.run(self.loss_hole, feed_in)))

    def generate_image(self, names, concat, save_dir='save'):
        if not self.built:
            self.build()
        images, masks, masked_imgs = self.data_feeder.fetch_data()
        ones = np.ones_like(masks)
        feed_in = {self.masked_image: masked_imgs, self.mask: masks,
                self.real_image: images, self.ones: ones}
        self.data_feeder.save_images(self.sess.run(self.comp, feed_in), names, concat, save_dir)
        self.data_feeder.save_images(self.sess.run(self.fake_image, feed_in), 'fake_img'+names[6:], concat, save_dir)
        self.data_feeder.save_images(images, 'ori_img'+names[6:], concat, save_dir)
        self.data_feeder.save_images(masked_imgs, 'masked_img'+names[6:], concat, save_dir)
        
    def save_models(self, name, save_dir='save'):
        self.net.save_models(name, save_dir)




