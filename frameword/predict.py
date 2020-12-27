import tensorflow as tf
import numpy as np
import logging


def create_sess():
    tf.reset_default_graph()
    sess = tf.Session()
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "./1")
    return sess


def predict(sess, data):
    # saved_model_cli show --dir ./1 --all
    in_feat = sess.graph.get_tensor_by_name("features:0")

    output = sess.graph.get_tensor_by_name("dense/BiasAdd:0")
    res = sess.run(output, feed_dict={in_feat: np.array(data)}, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
    return res


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("begin predict")
    sess = create_sess()
    data = np.random.random(size=[20, 2])
    i = 0
    batch_size = 5
    res = []
    while i < len(data):
        res.append(predict(sess, data[i:i+batch_size, :]))
        i += batch_size
    res = np.concatenate(res, axis=0)
    print(np.concatenate([data, res], axis=1))
    logging.info("finish predict")
