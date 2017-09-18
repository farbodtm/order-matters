import numpy as np
import resnet_model
import tensorflow as tf
import cPickle

def run(hps):

  image = tf.zeros([1, 32, 32, 3])
  labels = tf.zeros([1, 10])

  model = resnet_model.ResNet(hps, image, labels, 'eval')
  model.build_graph()
  saver = tf.train.Saver()

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  try:
    ckpt_state = tf.train.get_checkpoint_state('./out/normal')
    print ckpt_state
  except tf.errors.OutOfRangeError as e:
    tf.logging.error('Cannot restore checkpoint: %s', e)
    return
  print ckpt_state

  saver.restore(sess, ckpt_state.model_checkpoint_path)
  
  image_size = 32
  depth = 3
  image_bytes = image_size * image_size * depth

  def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

  re = {}
  correct = []
  batches = ['py-cifar/cifar-10-batches-py/data_batch_1', 'py-cifar/cifar-10-batches-py/data_batch_2', 'py-cifar/cifar-10-batches-py/data_batch_3', 'py-cifar/cifar-10-batches-py/data_batch_4', 'py-cifar/cifar-10-batches-py/data_batch_5']
  for fnames in batches:
    with open(fnames, 'rb') as f:
      dic = cPickle.load(f)
    for i in range(100):
      batch_size = 100
      data_images = dic['data'][(i*batch_size):((i+1)*batch_size)]
      data_labels = dic['labels'][i*batch_size:(i+1)*batch_size]
      image = tf.constant(data_images)
      image = tf.reshape(image, [batch_size,depth, image_size, image_size])
      image = tf.cast(tf.transpose(image, [0, 2, 3, 1]), tf.float32)
      image = tf.map_fn(lambda im : tf.image.per_image_standardization(im), image)
      image = tf.reshape(image, [batch_size, image_size, image_size, depth])

      labels = np.zeros([batch_size], dtype=int)
      labels = dense_to_one_hot(labels)

      images = tf.Session().run(image)
      (loss, predictions, train_step) = sess.run(
          [model.cost, model.predictions, model.global_step], feed_dict={model.images: images, model.labels: labels})
      predictions = np.argmax(predictions, axis=1)
      for i, p in enumerate(predictions):
        if predictions[i] != data_labels[i]:
          if not re.has_key(data_labels[i]):
            re[data_labels[i]] = []
          re[data_labels[i]].append(data_images[i])
        else:
          correct.append((data_labels[i], data_images[i]))

  count = {}
  fc = 0
  for c in count:
    fc += count[c]
  fil = np.array([])
  for cls in re:
    count[cls] = len(re[cls])
    la = np.array([int(cls)])
    for im in re[cls]:
      fil = np.concatenate((fil, la))
      fil = np.concatenate((fil, im))
  for im in range(fc):
    fil = np.concatenate((fil, correct[im][0]))
    fil = np.concatenate((fil, correct[im][1]))
  with open('./new_training.bin', 'wb') as f:
    f.write(fil.tostring())
      
def main(_):
  hps = resnet_model.HParams(batch_size=100,
                             num_classes=10,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')

  dev = '/cpu:0'
  with tf.device(dev):
    run(hps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

