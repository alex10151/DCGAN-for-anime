import tensorflow as tf
import tensorflow.contrib.slim as sl
import cv2
import os
import numpy as np
base_dir = './faces/'
counter = 0
x_inputs = []
for filename in os.listdir(base_dir):
    image = cv2.imread(base_dir+filename,cv2.IMREAD_COLOR)
    if image is not None:
        counter = counter+1
        print(counter)
        x_inputs.append(image.tolist())
        if(counter==1000):
            break
print(np.shape(x_inputs))


def generator(z, z_dim, initializer):
    gw1 = tf.get_variable('gw1_gen', [z_dim, 6 * 6 * 256], initializer=initializer,
                          regularizer=tf.contrib.layers.l2_regularizer(0.1))
    gb1 = tf.get_variable('gb1_gen', [6 * 6 * 256], initializer=tf.constant_initializer(0.0))
    gh1 = tf.contrib.layers.batch_norm(tf.matmul(z, gw1) + gb1)
    gh1 = tf.nn.relu(gh1)
    gh1 = tf.reshape(gh1, [-1, 6, 6, 256])
    # kernel_conv1 = tf.get_variable(name= 'filters_conv1_gen',shape= [5,5,64,256], initializer = initializer)
    conv1 = tf.layers.conv2d_transpose(gh1, 64, 5, strides=2, padding='SAME')
    conv1 = tf.contrib.layers.batch_norm(conv1)
    conv1 = tf.nn.relu(conv1)

    # kernel_conv2 = tf.get_variable(name= 'filters_conv2_gen',shape= [5,5,32,64], initializer = initializer)
    conv2 = tf.layers.conv2d_transpose(conv1, 32, 5, strides=2, padding='SAME')
    conv2 = tf.contrib.layers.batch_norm(conv2)
    conv2 = tf.nn.relu(conv2)

    # kernel_conv3 = tf.get_variable(name= 'filters_conv3_gen',shape= [5,5,16,32], initializer = initializer)
    conv3 = tf.layers.conv2d_transpose(conv2, 16, 5, strides=2, padding='SAME')
    conv3 = tf.contrib.layers.batch_norm(conv3)
    conv3 = tf.nn.relu(conv3)

    # kernel_conv4 = tf.get_variable(name= 'filters_conv4_gen',shape= [5,5,8,16], initializer = initializer)
    conv4 = tf.layers.conv2d_transpose(conv3, 8, 5, strides=2, padding='SAME')
    conv4 = tf.contrib.layers.batch_norm(conv4)
    conv4 = tf.nn.relu(conv4)

    # kernel_out = tf.get_variable(name= 'kernel_out_gen',shape= [96,96,3,8], initializer = initializer)
    out = tf.layers.conv2d_transpose(conv4, 3, 96, strides=1, padding='SAME')
    out = tf.nn.tanh(out)

    return out


def discriminator(x, initializer, reuse='False'):
    with tf.variable_scope('discriminator', reuse=reuse):
        kernel1 = tf.get_variable('filter1_dis', shape=[4, 4, 3, 16], initializer=initializer)
        conv1 = tf.nn.conv2d(x, kernel1, strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.contrib.layers.batch_norm(conv1)
        conv1 = tf.nn.relu(conv1)

        kernel2 = tf.get_variable('filter2_dis', shape=[4, 4, 16, 8], initializer=initializer)
        conv2 = tf.nn.conv2d(conv1, kernel2, strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.contrib.layers.batch_norm(conv2)
        conv2 = tf.nn.relu(conv2)

        kernel3 = tf.get_variable('filter3_dis', shape=[4, 4, 8, 4], initializer=initializer)
        conv3 = tf.nn.conv2d(conv2, kernel3, strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.contrib.layers.batch_norm(conv3)
        conv3 = tf.nn.relu(conv3)
        kernel4 = tf.get_variable('filter4_dis', shape=[4, 4, 4, 2], initializer=initializer)
        conv4 = tf.nn.conv2d(conv3, kernel4, strides=[1, 2, 2, 1], padding='SAME')
        conv4 = tf.contrib.layers.batch_norm(conv4)
        conv4 = tf.nn.relu(conv4)

        w = tf.get_variable('w_dis', shape=[72, 1], initializer=initializer)
        b = tf.get_variable('b_dis', shape=[1], initializer=tf.constant_initializer(0.0))
        out = tf.contrib.layers.flatten(conv4)
        out = tf.nn.sigmoid(tf.matmul(out, w) + b)

        return out
tf.reset_default_graph()

print('start integration:')
z_size = 100

initializer = tf.contrib.layers.xavier_initializer()

z_input = tf.placeholder(shape=[None, z_size], dtype=tf.float32)
image_input = tf.placeholder(shape=[None, 96,96,3], dtype=tf.float32)

gz = generator(z_input, z_size, initializer)
dz = discriminator(gz, initializer,None)
dx = discriminator(image_input,initializer, reuse=True )

loss_g = -tf.reduce_mean(tf.log(dz))
loss_d = -tf.reduce_mean(tf.log(dx) + tf.log(1.-dz))

opt_d = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)

opt_g = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)

var = tf.trainable_variables()

d_grads = opt_d.compute_gradients( loss_d, [v for v in var if 'dis' in v.name])
g_grads = opt_g.compute_gradients( loss_g, [v for v in var if 'gen' in v.name])

update_d = opt_d.apply_gradients(d_grads)
update_g = opt_g.apply_gradients(g_grads)

print('start to train:')
batch_size = 100
epoch = 20

gen_image_dir = './gen_image'
model_save_dir = './model_save'
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    #x_inputs = np.array(x_inputs)
    for i in range(epoch):
        batch = 0
        counter1 = 1
        print('epoch '+str(i)+'started:>>>>>>>>>>>')
        print(batch + batch_size)
        print(len(x_inputs))
        while ((batch + batch_size) < len(x_inputs)):
            print('batch'+str(counter1)+'has begun to train>>>>>>>>>>>>>')
            z_in = np.random.uniform(-1.0, 1.0, size=[batch_size, z_size]).astype(np.float32)
            lossd, _ = sess.run([loss_d, update_d],
                                feed_dict={z_input: z_in, image_input: x_inputs[batch:(batch + batch_size)]})
            lossg, _ = sess.run([loss_g, update_g], feed_dict={z_input: z_in})
            lossg, _ = sess.run([loss_g, update_g], feed_dict={z_input: z_in})
            batch = batch + batch_size
            counter1 = counter1 + 1
            print(lossd, lossg)

        print('batch' + str(counter1) + 'has begun to train>>>>>>>>>>>>>')
        z_in = np.random.uniform(-1.0, 1.0, size=[len(x_inputs) - batch, z_size]).astype(np.float32)
        lossd, _ = sess.run([loss_d, update_d], feed_dict={z_input: z_in, image_input: x_inputs[batch:]})
        lossg, _ = sess.run([loss_g, update_g], feed_dict={z_input: z_in})
        lossg, _ = sess.run([loss_g, update_g], feed_dict={z_input: z_in})
        print('last batch has finished.')
        print('No :' + str(i) + 'epoch' + str(loss_d) + '   ' + str( loss_g))
        if (epoch == 5):
            if (not os.path.exists(gen_image_dir)):
                os.makedirs(gen_image_dir)
            z = np.random.uniform(-1.0, 1.0, size=[5, z_size]).astype(np.float32)
            gen_img = sess.run(gz, feed_dict={z_input: z})
            for i, img in enumerate(gen_img):
                cv2.imwrite(gen_image_dir + '/' + str(epoch) + str(i) + '.jpg', img)
            if (not os.path.exists(model_save_dir)):
                os.makedirs(model_save_dir)
            saver.save(sess, model_save_dir + str(epoch) + '.cptk')