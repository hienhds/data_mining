import tensorflow as tf
from KGAT import KGAT
from Main import load_data_from_mysql
from argparse import Namespace

def load_model():
    data_config = load_data_from_mysql()
    args = Namespace(
        embed_size=384,
        kge_size=384,                     # có thể chọn bằng 1/2 hoặc 1/3 embed_size
        batch_size=256,
        batch_size_kg=128,
        lr=0.001,
        layer_size='[256, 128]',         # nên bắt đầu giảm dần
        alg_type='kgat',
        adj_type='norm',
        adj_uni_type='sum',
        regs='[1e-5, 1e-5]',
        verbose=1
    )

    model = KGAT(data_config=data_config, pretrain_data=None, args=args)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, "./kgat_model.ckpt")

    return model, sess, data_config