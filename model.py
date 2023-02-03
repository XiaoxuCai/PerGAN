from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import vgg16

from ops import *
from utils import *

class pix2pix(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=3, output_c_dim=1, dataset_name='Sal',
                 checkpoint_dir=None, sample_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        #self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn11 = batch_norm(name='d_bn11')
        self.d_bn12 = batch_norm(name='d_bn12')
        self.d_bn13 = batch_norm(name='d_bn13')

        self.d_bn21 = batch_norm(name='d_bn21')
        self.d_bn22 = batch_norm(name='d_bn22')

        self.d_bn31 = batch_norm(name='d_bn31')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')
        
        self.g_bn_s2 = batch_norm(name='g_bn_s2')
        self.g_bn_s3 = batch_norm(name='g_bn_s3')
        self.g_bn_s4 = batch_norm(name='g_bn_s4')
        self.g_bn_s5 = batch_norm(name='g_bn_s5')
        self.g_bn_s6 = batch_norm(name='g_bn_s6')
        self.g_bn_s7 = batch_norm(name='g_bn_s7')
        self.g_bn_s8 = batch_norm(name='g_bn_s8')
         
        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')
        self.g_bn_d8 = batch_norm(name='g_bn_d8')
        self.g_bn_d9 = batch_norm(name='g_bn_d9')

        
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        #self.real_A = self.real_data[:, :, :, :self.input_c_dim]/255.
        #self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]


        self.real_A = self.real_data[:, :, :, :3]/255
        self.real_A = tf.image.resize_images(self.real_A, [224, 224])
        #self.real_B = (self.real_data[:, :, :, 3:4]/361.4-1.)*0.9
        self.real_B = self.real_data[:, :, :, 3:4]/255
        self.real_B = tf.image.resize_images(self.real_B, [224, 224])
		

        self.fake_B = self.generator(self.real_A)
        self.real_AB = tf.concat([self.real_A, self.real_B],3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B],3)

        self.D1, self.D1_logits = self.discriminator1(self.real_AB, reuse=False)
        self.D1_, self.D1_logits_ = self.discriminator1(self.fake_AB, reuse=True)
        self.D2, self.D2_logits = self.discriminator2(self.real_AB, reuse=False)
        self.D2_, self.D2_logits_ = self.discriminator2(self.fake_AB, reuse=True)
        self.D3, self.D3_logits = self.discriminator3(self.real_AB, reuse = False)
        self.D3_, self.D3_logits_ = self.discriminator3(self.fake_AB, reuse=True)

        self.fake_B_sample =self.sampler(self.real_A)

        self.d_sum = tf.summary.histogram("d", (self.D1+self.D2+self.D3))
        self.d__sum = tf.summary.histogram("d_", (self.D1_+self.D2_+self.D3_))
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        self.d_loss_real1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D1_logits, labels = tf.ones_like(self.D1)))
        self.d_loss_fake1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D1_logits_, labels = tf.zeros_like(self.D1_)))

        self.d_loss_real2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D2_logits, labels = tf.ones_like(self.D2)))
        self.d_loss_fake2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D2_logits_, labels = tf.zeros_like(self.D2_)))

        self.d_loss_real3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D3_logits, labels = tf.ones_like(self.D3)))
        self.d_loss_fake3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D3_logits_, labels = tf.zeros_like(self.D3_)))

        self.d_loss_real = self.d_loss_real1 + self.d_loss_real2 + self.d_loss_real3
        self.d_loss_fake = self.d_loss_fake1 + self.d_loss_fake2 + self.d_loss_fake3

        self.content_loss = content_loss(self.real_AB, self.fake_AB)
        self.semantic_loss = semantic_loss(self.real_AB, self.fake_AB, 224*224)

        #self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_logits_, labels=tf.ones_like(self.D1_))) \
         #               + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))
        self.g_loss = (self.d_loss_real1+self.d_loss_real2+self.d_loss_real3) \
                         + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))\
						 + 100*(self.content_loss+ self.semantic_loss)

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def load_random_samples(self):
        data = np.random.choice(glob('./datasets/{}/val/*.mat'.format(self.dataset_name)), self.batch_size)
        sample = [load_cube(sample_file) for sample_file in data]

        #if (self.is_grayscale):
         #   sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        #else:
        sample_images = np.array(sample).astype(np.float32)
        return sample_images

    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples()
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images}
        )
        #save_cube(samples,
         #        './{}/train_{:02d}_{:04d}.mat'.format(sample_dir, epoch, idx))
        save_images(samples, [self.batch_size, 1],
                    './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def train(self, args):
        """Train pix2pix"""
        d_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(args.lr, beta1=args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum,
           self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(args.epoch):
            data = glob('./datasets/{}/train/*.mat'.format(self.dataset_name))
            #np.random.shuffle(data)
            batch_idxs = min(len(data), args.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [load_cube(batch_file) for batch_file in batch_files]
                #if (self.is_grayscale):
                 #   batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                #else:
                batch_images = np.array(batch).astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_loss_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_loss_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_loss_sum],
                                               feed_dict={ self.real_data: batch_images })
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                errG = self.g_loss.eval({self.real_data: batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, 500) == 2:
                    self.save(args.checkpoint_dir, counter)

    def discriminator1(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator1") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h10 = lrelu(conv2d(image, self.df_dim, name='d_h10_conv'))
            # h0 is (128 x 128 x self.df_dim)
            h11 = lrelu(self.d_bn11(conv2d(h10, self.df_dim*2, name='d_h11_conv')))
            # h1 is (64 x 64 x self.df_dim*2)
            h12 = lrelu(self.d_bn12(conv2d(h11, self.df_dim*4, name='d_h12_conv')))
            # h2 is (32x 32 x self.df_dim*4)
            h13 = lrelu(self.d_bn13(conv2d(h12, self.df_dim*8, d_h=1, d_w=1, name='d_h13_conv')))
            # h3 is (16 x 16 x self.df_dim*8)
            h14 = linear(tf.reshape(h13, [self.batch_size, -1]), 1, 'd_h13_lin')

            return tf.nn.sigmoid(h14), h14

    def discriminator2(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator2") as scope:

             if reuse:
                tf.get_variable_scope().reuse_variables()
             else:
                assert tf.get_variable_scope().reuse == False

             image2 = image[:,::2,::2,:]
             #image2 is 128*128*(input_c_dim + output_c_dim)
             h20 = lrelu(conv2d(image2, self.df_dim, name='d_h20_conv'))
             # h20 is (64 x 64 x self.df_dim)
             h21 = lrelu(self.d_bn21(conv2d(h20, self.df_dim*2, name='d_h21_conv')))
             # h1 is (32 x 32 x self.df_dim*2)
             h22 = lrelu(self.d_bn22(conv2d(h21, self.df_dim*4, name='d_h22_conv')))
             # h2 is (16x 16 x self.df_dim*4)
             h23 = linear(tf.reshape(h22, [self.batch_size, -1]), 1, 'd_h22_lin')

             return tf.nn.sigmoid(h23), h23

    def discriminator3(self, image, y=None, reuse = False):

        with tf.variable_scope("discriminator3") as scope:

            # image is 256 x 256 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            image3 = image[:,::4,::4,:]
             # image3 is (64 x 64 x self.df_dim)
            h30 = lrelu(conv2d(image3, self.df_dim, name='d_h30_conv'))
             # h30 is (32 x 32 x self.df_dim)
            h31 = lrelu(self.d_bn31(conv2d(h30, self.df_dim*2, name='d_h31_conv')))
             # h31 is (16 x 16 x self.df_dim*2)
            h32 = linear(tf.reshape(h31, [self.batch_size, -1]), 1, 'd_h31_lin')

            return tf.nn.sigmoid(h32), h32

    def generator(self, image, y=None):
        with tf.variable_scope("generator") as scope:
            
            vgg = vgg16.Vgg16()
            vgg.build(image)
            e2 = vgg.conv2_1
            e3 = vgg.conv3_1
            e4 = vgg.conv4_1
            e5 = vgg.conv5_1
            e6 = vgg.pool5

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(e6),
                [self.batch_size, 14, 14, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e5], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, 28, 28, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e4], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, 56, 56,self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e3], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, 112, 112, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e2], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)
            
            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, 224, 224, self.gf_dim], name='g_d8', with_w=True)
            d8 = self.g_bn_d8(self.d8)
                        
            self.d9, self.d9_w, self.d9_b = deconv2d(tf.nn.relu(d8),
                [self.batch_size, 224, 224, 32], k_h=1, k_w=1, d_h=1, d_w=1, name='g_d9', with_w=True)
            d9 = self.g_bn_d9(self.d9)
            
            self.d10, self.d10_w, self.d10_b = deconv2d(tf.nn.relu(d9),
                [self.batch_size, 224, 224, 1], k_h=1, k_w=1, d_h=1, d_w=1, name='g_d10', with_w=True)
          
            return tf.nn.sigmoid(self.d10)

    def sampler(self, image, y=None):

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            vgg = vgg16.Vgg16()
            vgg.build(image)
            e2 = vgg.conv2_1
            e3 = vgg.conv3_1
            e4 = vgg.conv4_1
            e5 = vgg.conv5_1
            e6 = vgg.pool5

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(e6),
                [self.batch_size, 14, 14, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e5], 3)
            # d4 is (16 x 16 x self.gf_dim*8*2)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, 28, 28, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e4], 3)
            # d5 is (32 x 32 x self.gf_dim*4*2)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, 56, 56,self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e3], 3)
            # d6 is (64 x 64 x self.gf_dim*2*2)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, 112, 112, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e2], 3)
            # d7 is (128 x 128 x self.gf_dim*1*2)
            
            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, 224, 224, self.gf_dim], name='g_d8', with_w=True)
            d8 = self.g_bn_d8(self.d8)
                        
            self.d9, self.d9_w, self.d9_b = deconv2d(tf.nn.relu(d8),
                [self.batch_size, 224, 224, 32], k_h=1, k_w=1, d_h=1, d_w=1, name='g_d9', with_w=True)
            d9 = self.g_bn_d9(self.d9)
            
            self.d10, self.d10_w, self.d10_b = deconv2d(tf.nn.relu(d9),
                [self.batch_size, 224, 224, 1], k_h=1, k_w=1, d_h=1, d_w=1, name='g_d10', with_w=True)
          
            return tf.nn.sigmoid(self.d10)

    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test pix2pix"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        sample_files = glob('./datasets/{}/test/*.mat'.format(self.dataset_name))

        # sort testing input
        #n = [int(i) for i in map(lambda x: x.split('\\')[-1].split('.mat')[0], sample_files)]
        #sample_files = [x for (y, x) in sorted(zip(n, sample_files))]

        # load testing input
        print("Loading testing images ...")
        sample = [load_cube(sample_file, is_test=True) for sample_file in sample_files]

       # if (self.is_grayscale):
        #    sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        #else:
        sample_images = np.array(sample).astype(np.float32)

        sample_images = [sample_images[i:i+self.batch_size]
                         for i in xrange(0, len(sample_images), self.batch_size)]
        sample_images = np.array(sample_images)
        print(sample_images.shape)

        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i, sample_image in enumerate(sample_images):
            idx = i+1
            print("sampling image ", idx)
            samples = self.sess.run(
                self.fake_B_sample,
                feed_dict={self.real_data: sample_image}
            )
            sample_file = sample_files[idx-1]
            #save_cube(samples,
             #           './{}/test_{:04d}.mat'.format(args.test_dir, idx))
            save_images(samples, [self.batch_size, 1],
                        './{}/{}.png'.format(args.test_dir, sample_file[20:-4]))
