import tensorflow as tf
import os
import logging
import numpy as np

model_files_base = './model_files/'
tfrecord_path_base = './tfrecord/'


class Model:
    def inference(self, dense_feature):
        output = tf.layers.dense(dense_feature, 1)
        return output
    def loss(self, pred, label):
        return tf.abs(pred - label)

def parse_example(example):
    dic = {
        "x": tf.FixedLenFeature(dtype=tf.float32, shape=[2]),
        "y": tf.FixedLenFeature(dtype=tf.float32, shape=[1])
    }
    feature = tf.parse_single_example(example, dic)
    return feature["x"], feature["y"]

def read_tfrecord():
    file_list = list(map(lambda x: os.path.join(tfrecord_path_base, x), os.listdir(tfrecord_path_base)))
    data_set = tf.data.TFRecordDataset(file_list, num_parallel_reads=4)
    data_set = data_set.map(parse_example, num_parallel_calls=4)
    data_set = data_set.batch(batch_size=32, drop_remainder=True)
    return data_set.make_one_shot_iterator()


def train_process():
    if not os.path.exists(model_files_base):
        os.popen("mkdir {}".format(model_files_base))
    data = read_tfrecord()
    x, y = data.get_next()
    model = Model()
    output = model.inference(x)
    loss = tf.reduce_mean(model.loss(output, y))
    opt = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    variables = tf.trainable_variables()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            try:
                _, loss_value = sess.run([opt, loss])
                if step % 100 == 0:
                    print(step, loss_value)
                step += 1
            except tf.errors.OutOfRangeError:
                break
        logging.info("train done")
        # variables
        variables_res = sess.run(variables)
        for v1, v2 in zip(variables, variables_res):
            print(v1, v2)

        # save
        saver = tf.train.Saver()
        saver.save(sess, save_path=os.path.join(model_files_base, "model"), global_step=step)

    saver_model()


def saver_model():
    tf.reset_default_graph()
    model = Model()
    features = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="features")
    output = model.inference(features)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, tf.train.latest_checkpoint(model_files_base))
        tensor_feature = tf.saved_model.build_tensor_info(features)
        tensor_output = tf.saved_model.build_tensor_info(output)

        if os.path.exists('./1'):
            os.popen("rm -r ./1")

        builder = tf.saved_model.builder.SavedModelBuilder("./1")
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={"features": tensor_feature},
                outputs={"output": tensor_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            )
        legacy_init_op = tf.group([tf.tables_initializer(), tf.local_variables_initializer()], name='legacy_init_op')
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={"predict": prediction_signature},
            legacy_init_op=legacy_init_op
        )
        builder.save()


if __name__ == '__main__':
    train_process()

