# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet Train/Eval module.
"""
import time
import six
import sys
import collections

import cifar_input
import numpy as np
import order_model
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dataset', 'cifar10', 'cifar10 or cifar100.')
tf.app.flags.DEFINE_integer('num_cls', 10, 'number of classes.')
tf.app.flags.DEFINE_string('mode', 'train', 'train or eval.')

tf.app.flags.DEFINE_string('start_model_path', '',
                           'Model path to restore the variable from.')
tf.app.flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
tf.app.flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
tf.app.flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
tf.app.flags.DEFINE_integer('eval_batch_count', 139,
                            'Number of batches to eval.')
tf.app.flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')

def lookup_table(n):
  table = []
  arr = range(n)
  table.append(arr)
  for i in range(n-1):
    new_arr = list(arr)
    new_arr.insert(0, new_arr.pop())
    table.append(new_arr)
    arr = new_arr
  return table

def train(hps):
  """Training loop."""
  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode)
  model = order_model.ResNet(hps, images, labels, FLAGS.mode)
  model.build_graph()

  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.
          TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

  tf.contrib.tfprof.model_analyzer.print_model_analysis(
      tf.get_default_graph(),
      tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

  truth = tf.argmax(model.labels, axis=1)
  predictions = tf.argmax(model.predictions, axis=1)
  permutations = tf.argmax(model.permutations, axis=1)
  table = tf.constant(lookup_table(FLAGS.num_cls), dtype=tf.int64)
  def map_label_fn_p(i):
    pred = predictions[i]
    perm = permutations[i]
    return table[perm][pred]
  final_predictions = tf.map_fn(map_label_fn_p, tf.range(128, dtype=tf.int64), dtype=tf.int64)
  #final_perdictions = tf.argmax(new_predictions, axis=1)

  precision = tf.reduce_mean(tf.to_float(tf.equal(final_predictions, truth)))

  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=FLAGS.train_dir,
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('Precision', precision)]))

  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.cost,
               'precision': precision},
      every_n_iter=100)

  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 0.1

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step,  # Asks for global step value.
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      train_step = run_values.results
      if train_step < 40000:
        self._lrn_rate = 0.1
      elif train_step < 80000:
        self._lrn_rate = 0.01
      elif train_step < 100000:
        self._lrn_rate = 0.001
      else:
        exit()
      #if train_step < 40000:
      #  self._lrn_rate = 0.01
      #elif train_step < 60000:
      #  self._lrn_rate = 0.01
      #elif train_step < 80000:
      #  self._lrn_rate = 0.001
      #else:
      #  exit()
        #self._lrn_rate = 0.0001

  variables = tf.all_variables()
  variables = filter(lambda x: not x.name.startswith('global_step'), variables)
  print variables

  if (FLAGS.start_model_path):
    restore_fn = tf.contrib.framework.assign_from_checkpoint_fn(FLAGS.start_model_path, variables, ignore_missing_vars=True)
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      save_checkpoint_secs=200,
      config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
    if (FLAGS.start_model_path):
      restore_fn(mon_sess)
    while not mon_sess.should_stop():
    #for i in range(100):
      mon_sess.run(model.train_op)


def evaluate(hps):
  """Eval loop."""
  images, labels = cifar_input.build_input(
      FLAGS.dataset, FLAGS.eval_data_path, hps.batch_size, FLAGS.mode)

  model = order_model.ResNet(hps, images, labels, FLAGS.mode)
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)

  best_precision = 0.0
  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    total_prediction, correct_prediction = 0, 0
    truth_count = collections.Counter()
    label_count = collections.Counter()
    perm_count = collections.Counter()
    for _ in six.moves.range(FLAGS.eval_batch_count):
      (summaries, loss, predictions, permutations, truth, train_step) = sess.run(
          [model.summaries, model.cost, model.predictions, model.permutations,
           model.labels, model.global_step])

      truth = np.argmax(truth, axis=1)
      n_predictions = np.argmax(predictions, axis=1)
      permutations = np.argmax(permutations, axis=1)

      truth_count += collections.Counter(truth)
      label_count += collections.Counter(n_predictions)
      perm_count += collections.Counter(permutations)

      predictions = []
      table = lookup_table(FLAGS.num_cls)
      for i in range(hps.batch_size):
        pred = n_predictions[i]
        perm = permutations[i]
        predictions.append(table[perm][pred])
      predictions = np.array(predictions)

      #final_perdictions = tf.argmax(new_predictions, axis=1)

      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]

    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)
    tf.logging.info('truth count:')
    print truth_count

    tf.logging.info('label count:')
    print label_count

    tf.logging.info('permutation count:')
    print perm_count

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, train_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag='Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, train_step)
    summary_writer.add_summary(summaries, train_step)
    tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                    (loss, precision, best_precision))

    summary_writer.flush()

    if train_step > 97000:
      break

    if FLAGS.eval_once:
      break

    time.sleep(200)


def main(_):
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  if FLAGS.mode == 'train':
    batch_size = 128
  elif FLAGS.mode == 'eval':
    batch_size = 1

  if FLAGS.dataset == 'cifar10':
    num_classes = 10
  elif FLAGS.dataset == 'cifar100':
    num_classes = 100

  hps = order_model.HParams(batch_size=batch_size,
                             num_classes=num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='mom')

  with tf.device(dev):
    if FLAGS.mode == 'train':
      train(hps)
    elif FLAGS.mode == 'eval':
      evaluate(hps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
