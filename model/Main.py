import mysql.connector
import scipy.sparse as sp
import tensorflow as tf
import numpy as np
from KGAT import KGAT
from argparse import Namespace
import random


def load_data_from_mysql():
    # 1. Kết nối MySQL
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="231123",
        database="cv"
    )
    cursor = conn.cursor(dictionary=True)

    # 2. Đọc nodes và embedding
    cursor.execute("SELECT id, node_type, embedding FROM nodes")
    node_rows = cursor.fetchall()
    
    node2idx = {}
    type_dict = {}
    embedding_list = []

    for idx, row in enumerate(node_rows):
        node_id = row['id']
        node2idx[node_id] = idx
        type_dict[node_id] = row['node_type']

        # Decode BLOB (assumed float32 vector)
        vec = np.frombuffer(row['embedding'], dtype=np.float32)
        embedding_list.append(vec)

    # Stack all embeddings into a matrix
    embedding_matrix = np.vstack(embedding_list).astype(np.float32)

    # Xác định các node là candidate và position
    user_ids = [nid for nid in type_dict if type_dict[nid] == 'candidate']
    job_ids = [nid for nid in type_dict if type_dict[nid] == 'job']
    n_users = len(user_ids)
    n_jobs = len(job_ids)
    n_nodes = len(node_rows)

    # Mapping user/job ids sang index trong embedding_matrix
    user_embed = embedding_matrix[[node2idx[nid] for nid in user_ids]]
    job_embed = embedding_matrix[[node2idx[nid] for nid in job_ids]]

    # Các node còn lại (non-job entities)
    job_indices = set(node2idx[nid] for nid in job_ids)
    entity_indices = [i for i in range(n_nodes) if i not in job_indices]
    entity_embed = embedding_matrix[entity_indices]

    # 3. Đọc relations và embedding
    cursor.execute("SELECT id, relation_name, embedding FROM relations")
    rel_rows = cursor.fetchall()

    rel2id = {}
    relation_embed_list = []
    for row in rel_rows:
        rel2id[row['id']] = row['relation_name']
        vec = np.frombuffer(row['embedding'], dtype=np.float32)
        relation_embed_list.append(vec)

    relation_embed = np.vstack(relation_embed_list).astype(np.float32)
    n_relations = len(rel2id)

    # 4. Đọc edges
    cursor.execute("SELECT head_node_id, relation_id, tail_node_id FROM edges")
    edge_rows = cursor.fetchall()

    all_h_list, all_r_list, all_t_list = [], [], []
    row_idx, col_idx, edge_val = [], [], []

    for row in edge_rows:
        h_idx = node2idx[row['head_node_id']]
        t_idx = node2idx[row['tail_node_id']]
        r_id = row['relation_id'] - 1

        all_h_list.append(h_idx)
        all_r_list.append(r_id)
        all_t_list.append(t_idx)

        row_idx.append(h_idx)
        col_idx.append(t_idx)
        edge_val.append(1.0)

    # 5. Xây ma trận A_in
    A = sp.coo_matrix(
        (edge_val + edge_val, (row_idx + col_idx, col_idx + row_idx)),
        shape=(n_nodes, n_nodes), dtype=np.float32
    )

    # 6. Trả về data_config
    return {
        'n_jobs': n_jobs,
        'n_users': n_users,
        'n_entities': len(entity_embed),
        'n_relations': n_relations,
        'A_in': A,
        'job_embed': job_embed,
        'user_embed': user_embed,
        'entity_embed': entity_embed,
        'relation_embed': relation_embed,
        'all_h_list': all_h_list,
        'all_r_list': all_r_list,
        'all_t_list': all_t_list,
        'all_v_list': [1.0] * len(all_h_list)
    }

def generate_cf_batches(data_config, batch_size):
    n_jobs = data_config['n_jobs']
    n_users = data_config['n_users']

    job_ids = list(range(n_jobs))
    user_ids = list(range(n_users))

    batches = []
    for _ in range(n_jobs // batch_size + 1):
        batch_jobs = random.sample(job_ids, batch_size)
        pos_users = random.choices(user_ids, k=batch_size)
        neg_users = random.choices(user_ids, k=batch_size)

        batches.append({
            'job_ids': batch_jobs,
            'pos_user_ids': pos_users,
            'neg_user_ids': neg_users
        })
    return batches

def generate_kg_batches(data_config, batch_size):
    h_list = data_config['all_h_list']
    r_list = data_config['all_r_list']
    t_list = data_config['all_t_list']
    n_entities = data_config['n_entities'] + data_config['n_jobs']

    triples = list(zip(h_list, r_list, t_list))
    random.shuffle(triples)

    batches = []
    for i in range(0, len(triples), batch_size):
        batch = triples[i:i + batch_size]
        h, r, pos_t = zip(*batch)
        neg_t = random.choices(range(n_entities), k=len(batch))

        batches.append({
            'h': list(h),
            'r': list(r),
            'pos_t': list(pos_t),
            'neg_t': neg_t
        })
    return batches


def main():
    # 1. Load dữ liệu từ MySQL → chuẩn hóa thành data_config
    data_config = load_data_from_mysql()

    # 2. Khởi tạo args (tham số mô hình)
    embed_size = 384

    args = Namespace(
        embed_size=embed_size,
        kge_size=embed_size,                     # có thể chọn bằng 1/2 hoặc 1/3 embed_size
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


    # 3. Khởi tạo model
    model = KGAT(data_config=data_config, pretrain_data=None, args=args)

    # 4. Tạo session và khởi tạo biến
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, 51):
            print(f"\nEpoch {epoch}")

            # --------- PHASE I: Collaborative Filtering ---------
            for cf_batch in generate_cf_batches(data_config, args.batch_size):
                feed_dict = {
                    model.jobs: cf_batch['job_ids'],
                    model.pos_users: cf_batch['pos_user_ids'],
                    model.neg_users: cf_batch['neg_user_ids'],
                    model.node_dropout: [0.1] * len(eval(args.layer_size)),
                    model.mess_dropout: [0.1] * len(eval(args.layer_size))
                }
                _, loss, base_loss, kge_loss, reg_loss = model.train(sess, feed_dict)

            # --------- PHASE II: Knowledge Graph Embedding ---------
            for kg_batch in generate_kg_batches(data_config, args.batch_size_kg):
                feed_dict = {
                    model.h: kg_batch['h'],
                    model.r: kg_batch['r'],
                    model.pos_t: kg_batch['pos_t'],
                    model.neg_t: kg_batch['neg_t']
                }
                _, kg_total_loss, kge_loss2, reg_loss2 = model.train_A(sess, feed_dict)

            # --------- Cập nhật attention matrix từ KG ---------
            model.update_attentive_A(sess)

            print(f"[Epoch {epoch}] CF Loss: {float(loss):.4f}, KG Loss: {float(kg_total_loss):.4f}")


        print("Training done.")

        # 5. Lưu mô hình
        saver = tf.train.Saver()
        saver.save(sess, "saved_model/kgat_model.ckpt")
        print("✅ Model saved compelete")


if __name__ == "__main__":
    main()
