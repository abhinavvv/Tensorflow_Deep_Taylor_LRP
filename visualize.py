import tensorflow as tf
import numpy as np
import sys, os, shutil, signal, time
import threading
import argparse, json
from readers.video_readers import ImageReader
from PIL import Image

tf.logging.set_verbosity(tf.logging.INFO)

print "TensorFlow version:", tf.__version__

slim = tf.contrib.slim
sys.path.append('base_models/tf-slim-models/slim')
from nets import inception

models_dir = "trained_models"
model_name1 = "inception-flex-crack-v3"
test_image_list_path = '/data01/patents/artificial_feature_saliency/scott/generated_test_images/generated.10K.txt'
sal_image_dest_dir = '/data01/patents/artificial_feature_saliency/scott/generated_test_images/generated.10K.sal/'

#model_name2 = "inception-flex-poor-solder-05-19-lr-jitter"
#model_name3 = "inception-flex-broken-cell-5-24"

#SC_THRESHOLD = 30

do_rename = False #True #TODO make this a switch

# MODEL PATHS - note that you have to rename scopes initially to be able to load all vars in the final combined graph (do_rename = True one time)
if do_rename:
  ckpt_dir1 = "{}/{}".format(models_dir, model_name1)
  ckpt_dir2 = "{}/{}".format(models_dir, model_name2)
  ckpt_dir3 = "{}/{}".format(models_dir, model_name3)
else:
  ckpt_dir1 = "{}/crack".format(models_dir)
  ckpt_dir2 = "{}/poor_solder".format(models_dir)
  ckpt_dir3 = "{}/broken_cell".format(models_dir)

models_map = {
  ckpt_dir1: "crack",
#  ckpt_dir2: "poor_solder",
#  ckpt_dir3: "broken_cell",
#  "dummy": "short_circuit"
}

model_names = ["crack"]#, "poor_solder", "broken_cell", "short_circuit"]
ordered_ckpt_dirs = [ckpt_dir1]#,ckpt_dir2,ckpt_dir3,"dummy"]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
cfg = tf.ConfigProto(gpu_options=gpu_options)


# build pipeline
def build_graph():
  images_ph = tf.placeholder(tf.float32, shape=[None,299,299,3])
  labels_ph = tf.placeholder(tf.int32, shape=[None,])
  with slim.arg_scope(inception.inception_v3_arg_scope()):
    l, _ = inception.inception_v3(images_ph, num_classes=2, is_training=False, reuse=False, scope="crack/InceptionV3")

  # Construct the scalar neuron tensor.
  logits = tf.get_default_graph().get_tensor_by_name('crack/InceptionV3/Logits/SpatialSqueeze:0')
  neuron_selector = tf.placeholder(tf.int32)
  y = logits[0][neuron_selector]

  # Construct tensor for predictions.
  prediction = tf.argmax(logits, 1)
  probs = tf.nn.softmax(l)
  return images_ph, prediction, probs, y, neuron_selector

def restore_variables(session=None):
  for ckpt_dir in models_map:
    if ckpt_dir == "dummy":
      continue
    prefix = models_map[ckpt_dir]
    saver = tf.train.Saver([v for v in tf.global_variables() if v.name.startswith(prefix)])
    saver.restore(session, tf.train.latest_checkpoint(ckpt_dir))
    print "RESTORED", prefix

def SaveImage(im, title='', ax=None):
  im = ((im + 1) * 127.5).astype(np.uint8)
  P.imshow(im)
  P.title(title)

def SaveGrayscaleImage(im, title='', ax=None):
  P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
  P.title(title)


def LoadImage(file_path):
  im = PIL.Image.open(file_path)
  # scale to -1.0 to +1.0
  im = np.asarray(im)
  return im/127.5 - 1.0

def prepare_image_graph():
  path_placeholder = tf.placeholder(tf.string)
  image_path_tensor = tf.read_file(path_placeholder)
  image_tensor = tf.image.decode_jpeg(image_path_tensor, channels=3)
  resized_image_tensor = tf.image.resize_images(image_tensor, [299, 299]) #bilinear
  # move all values to [-1.0, +1.0] interval for inception
  scaled_image_tensor = tf.image.convert_image_dtype(resized_image_tensor, dtype=tf.float32) # scales to [0.0, 1.0]
  scaled_image_tensor = tf.subtract(scaled_image_tensor, 0.5) 
  scaled_image_tensor = tf.multiply(scaled_image_tensor, 2.0) # move to [-1.0, 1.0]
  return path_placeholder, resized_image_tensor, scaled_image_tensor

def setup_graph(session=None):
  images_ph, prediction, probs, y, neuron_selector = build_graph()
  restore_variables(session=sess)
  return images_ph, probs, y, prediction, neuron_selector

def predict(session, paths, images_ph, probabilities):
  path_placeholder, image_tensor, scaled_image_tensor = prepare_image_graph()
  image_batch = []
  scaled_image_batch = []

  for p in paths:
    scaled_img = sess.run(scaled_image_tensor, feed_dict={path_placeholder: p})
    scaled_image_batch.append(scaled_img)
    img = sess.run(image_tensor, feed_dict={path_placeholder: p})
    image_batch.append(img)
  probs = sess.run(probabilities, feed_dict={images_ph:scaled_image_batch})
  return probs, image_batch


def visualize_images(sess, images_ph, images, y, predicted_class, neuron_selector, image_names):

  visualize(sess, images_ph, images, y, predicted_class, neuron_selector, image_names)

  # for image, image_name in zip(images, image_names):
  #   visualize(sess, images_ph, image, y, predicted_class, neuron_selector, image_name)

#DEBUG - TESTING EXISTING LIST SUPPORT
def visualize(sess, images_ph, im, y, prediction_class, neuron_selector, base_name):
  import saliency
  graph = tf.get_default_graph()
  # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
  gradient_saliency = saliency.GradientSaliency(graph, sess, y, images_ph)
  guided_backprop = saliency.GuidedBackprop(graph, sess, y, images_ph)
  algo = guided_backprop
  # Compute the vanilla mask and the smoothed mask.

  for i,image in enumerate(im):
    prediction_class =  1 #np.argmax(prediction_class[i])
    vanilla_mask_3d = algo.GetMask(image, feed_dict = {neuron_selector: prediction_class})
    #smoothgrad_mask_3d = algo.GetSmoothedMask(im, feed_dict = {neuron_selector: prediction_class})

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
    #smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
    vanilla_mask_grayscale = ((vanilla_mask_grayscale + 1) * 127.5).astype(np.uint8)
    #smoothgrad_mask_grayscale = ((smoothgrad_mask_grayscale + 1) * 127.5).astype(np.uint8)
    result = Image.fromarray(vanilla_mask_grayscale, mode='L')
    #smoothed_result = Image.fromarray(smoothgrad_mask_grayscale, mode='L')
    #print 'sal name', base_name[i]
    print 'sal_image_dest_dir', sal_image_dest_dir
    sal_name =  sal_image_dest_dir + str(base_name[i]).replace('.jpg', '_sal.jpg')
    print 'saving {} ...'.format(sal_name)
    result.save(sal_name)
    #
    # print 'saving smooth__result.jpg ...'
    # smoothed_result.save('smooth__result.jpg')



# def visualize(sess, images_ph, im, y, prediction_class, neuron_selector, base_name):
#   import saliency
#   graph = tf.get_default_graph()
#   # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
#   gradient_saliency = saliency.GradientSaliency(graph, sess, y, images_ph)
#   guided_backprop = saliency.GuidedBackprop(graph, sess, y, images_ph)
#   algo = guided_backprop
#   # Compute the vanilla mask and the smoothed mask.
#   vanilla_mask_3d = algo.GetMask(im, feed_dict = {neuron_selector: prediction_class})
#   #smoothgrad_mask_3d = algo.GetSmoothedMask(im, feed_dict = {neuron_selector: prediction_class})
#
#   # Call the visualization methods to convert the 3D tensors to 2D grayscale.
#   vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
#   #smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
#   vanilla_mask_grayscale = ((vanilla_mask_grayscale + 1) * 127.5).astype(np.uint8)
#   #smoothgrad_mask_grayscale = ((smoothgrad_mask_grayscale + 1) * 127.5).astype(np.uint8)
#   result = Image.fromarray(vanilla_mask_grayscale, mode='L')
#   #smoothed_result = Image.fromarray(smoothgrad_mask_grayscale, mode='L')
#   print 'sal_image_dest_dir', sal_image_dest_dir
#   sal_name =  sal_image_dest_dir + str(base_name).replace('.jpg', '_sal.jpg')
#   print 'saving {} ...'.format(sal_name)
#   result.save(sal_name)
#   #
#   # print 'saving smooth__result.jpg ...'
#   # smoothed_result.save('smooth__result.jpg')

if __name__ == "__main__":

  with open(test_image_list_path) as f:
    test_image_list_full = f.readlines()
    test_image_list_full = [x.strip() for x in test_image_list_full]

  #TODO - need to break into batches
  #test_image_list = test_image_list[:10]

  batch_size = 300

  start = -1
  end = 500

  for x in range(500, 10000, batch_size):
    if x > end:
      start = end
      end = x
      print start, end

      test_image_list = test_image_list_full[start:end]

      image_names = []
      for image_name in test_image_list:
        base_name = image_name[str(image_name).rfind('/'):]
        image_names.append(base_name)

      sess = tf.Session(config=cfg)
      images_ph, probabilities, y, prediction, neuron_selector = setup_graph(session=sess)
      r,images = predict(sess, test_image_list, images_ph, probabilities)
      #predicted_class =  np.argmax(r[0])
      predicted_class = r
      #predicted_class = 1  # np.argmax(r[0])
      #im = images[0]
      #visualize(sess, images_ph, im, y, predicted_class, neuron_selector)
      visualize_images(sess, images_ph, images, y, predicted_class, neuron_selector, image_names)
