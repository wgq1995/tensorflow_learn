import tensorflow as tf
import os
import pandas as pd
import numpy as np


dataframe_path = "data.csv"
tfrecord_path_base = './tfrecord/'


def create_df():
    sample_n = 100000
    x1 = np.random.random(size=sample_n)
    x2 = np.random.random(size=sample_n)
    y = x1 * 2 + x2 * 3 + np.random.random(size=sample_n) / 10
    df = pd.DataFrame({'x1': x1, "x2": x2, "y": y})
    df.to_csv(dataframe_path, index_label=False)


def create_tfrecord():
    if not os.path.exists(dataframe_path):
        create_df()
    if not os.path.exists(tfrecord_path_base):
        os.popen("mkdir {}".format(tfrecord_path_base))
    chunk_size = 10000
    iter_df = pd.read_csv(dataframe_path, chunksize=chunk_size)
    i = 0
    for df in iter_df:
        writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_path_base, "part{}.tfrecord".format(i)))
        for row in df.itertuples():
            x = row[1:3]
            y = row[3:]
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "x": tf.train.Feature(float_list=tf.train.FloatList(value=x)),
                        "y": tf.train.Feature(float_list=tf.train.FloatList(value=y))
                    }
                )
            )
            writer.write(example.SerializeToString())
        writer.close()
        i += 1


if __name__ == '__main__':
    create_tfrecord()
