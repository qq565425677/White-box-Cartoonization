import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import network
import guided_filter
from tqdm import tqdm
from video_helper import is_video_file, video_to_frames, frames_to_video
import shutil

tf.disable_v2_behavior()


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def _model_init(model_path):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    return input_photo, final_out, sess


def _convert_video(input_photo, final_out, sess, load_folder, save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    name_list = os.listdir(load_folder)
    for name in tqdm(name_list):
        if not os.path.isfile(os.path.join(load_folder, name)):
            continue
        try:
            if is_video_file(name):
                tem_load_folder = os.path.join(load_folder, name[:-4])
                tem_save_folder = os.path.join(save_folder, name[:-4])
                video_to_frames(os.path.join(load_folder, name), tem_load_folder)
                _convert_video(input_photo, final_out, sess, tem_load_folder, tem_save_folder)
                frames_to_video(tem_save_folder, os.path.join(save_folder, name))
                shutil.rmtree(tem_load_folder)
                shutil.rmtree(tem_save_folder)

            load_path = os.path.join(load_folder, name)
            save_path = os.path.os.path.join(save_folder, name)
            image = cv2.imread(load_path)
            image = resize_crop(image)
            batch_image = image.astype(np.float32) / 127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            output = sess.run(final_out, feed_dict={input_photo: batch_image})
            output = (np.squeeze(output) + 1) * 127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, output)
        except:
            print('cartoonize {} failed'.format(load_path))


def cartoonize(load_folder, save_folder, model_path):
    input_photo, final_out, sess = _model_init(model_path)
    _convert_video(input_photo, final_out, sess, load_folder, save_folder)


if __name__ == '__main__':
    model_path = 'saved_models'
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    cartoonize(load_folder, save_folder, model_path)
