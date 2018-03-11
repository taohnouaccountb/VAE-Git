# ENVIRONMENT ARGUMENT
RUN_ON_CRANE = False
USE_EARLY_STOPPING = True
SAVE_MODEL = False
GPU_ID = '0'

import os
import tensorflow as tf
import numpy as np
from util import PSNR
from model import encoder_bundle, decoder_bundle
if not RUN_ON_CRANE:
    import matplotlib.pyplot as plt


flags = tf.app.flags
if RUN_ON_CRANE:
    flags.DEFINE_string('data_dir', '/work/cse496dl/shared/hackathon/05/', '')
    flags.DEFINE_string('save_dir', '/work/cse496dl/tyao/output/', '')
else:
    flags.DEFINE_string('data_dir', 'G:\\CIFAR\\', '')
    flags.DEFINE_string('save_dir', '.\\output\\', 'directory where model graph and weights are saved')

flags.DEFINE_integer('batch_size', 21, '')
flags.DEFINE_integer('max_epoch_num', 20, '')
flags.DEFINE_integer('patience', 4, '')
flags.DEFINE_float('REG_COEFF', 0.0, '')
FLAGS = flags.FLAGS


def main(argv):
    # 4×4-Fold
    train_x_1 = np.load(FLAGS.data_dir + 'cifar10_train_data.npy')
    train_x_1 = train_x_1.reshape([-1, 32, 32, 3])
    test_x_1 = np.load(FLAGS.data_dir + 'cifar10_test_data.npy')
    test_x_1 = test_x_1.reshape([-1, 32, 32, 3])
    # Retrieve properties of data
    num_train = train_x_1.shape[0]
    num_vali = test_x_1.shape[0]
    # Define Variables and Layers
    train_flag = tf.Variable(True)
    x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_placeholder')
    latent = encoder_bundle(x,training=train_flag)
    output = decoder_bundle(latent,training=train_flag)

    # Define enc-dec loss
    PSNR_loss = PSNR(x, output)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = -PSNR_loss + FLAGS.REG_COEFF * sum(regularization_losses)

    # Set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
    saver = tf.train.Saver()

    # Start Training
    config = tf.ConfigProto()
    if not RUN_ON_CRANE:
        config.gpu_options.visible_device_list = GPU_ID
    with tf.Session(config=config) as session:
        print('Variables in the graph:', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        session.run(tf.global_variables_initializer())
        test_accu_list = []
        min_test_val_index = 0
        batch_size = FLAGS.batch_size
        early_count = 0
        for epoch in range(FLAGS.max_epoch_num):
            print('\nEpoch: ' + str(epoch))

            # train
            train_flag.assign(True)
            ce_vals = []
            for i in range(num_train // batch_size):
                batch_xs = train_x_1[i * batch_size:(i + 1) * batch_size, :]
                _, train_ce = session.run([train_op, total_loss], {x: batch_xs})
                ce_vals.append(train_ce)
            avg_train_ce = sum(ce_vals) / len(ce_vals)  # after each fold , train err store in kfd_train_err

            # validate
            train_flag.assign(False)
            ce_vals = []
            for i in range(num_vali // FLAGS.batch_size):
                batch_xs = test_x_1[i * batch_size:(i + 1) * batch_size, :]
                vali_ce = session.run(PSNR_loss, {x: batch_xs})
                ce_vals.append(vali_ce)

            # analyse
            avg_vali_ce = sum(ce_vals) / len(ce_vals)
            print('TRAIN PSNR LOSS: ' + str(avg_train_ce))
            print('TEST PSNR LOSS: ' + str(avg_vali_ce))
            test_accu_list.append(avg_vali_ce)

            # early stopping
            if USE_EARLY_STOPPING:
                if len(test_accu_list) > 2:
                    if test_accu_list[-1] <= test_accu_list[min_test_val_index]:
                        early_count = early_count + 1
                    else:
                        early_count = 0
                        min_test_val_index = len(test_accu_list) - 1
                        if SAVE_MODEL:
                            file_name = "homework_3-0_{ep}_{rate}_{rate2}".format(ep=epoch,
                                                                                  rate=int(test_accu_list[-1] * 100),
                                                                                  rate2=int(
                                                                                      test_accu_list[-1] * 10000 % 100))
                            saver.save(session, os.path.join(FLAGS.save_dir, file_name),
                                       global_step=global_step_tensor)
                if early_count > FLAGS.patience:
                    break
        xx=train_x_1[-1:]
        output=session.run(output,{x:xx})
        if not RUN_ON_CRANE:
            imgplot = plt.imshow(xx[0])
            plt.show()
            imgplot = plt.imshow(output[0])
            plt.show()
    print("\nEND")


if __name__ == "__main__":
    tf.app.run()
