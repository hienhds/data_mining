import json
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from KGAT import KGAT
import mysql.connector
from argparse import Namespace
import mysql.connector
import scipy.sparse as sp

# Đảm bảo load_data_from_mysql trả thêm job2cf, user2cf
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

    # Tìm số lượng node
    max_id = max(row['id'] for row in node_rows)
    embedding_dim = len(np.frombuffer(node_rows[0]['embedding'], dtype=np.float32))
    embedding_matrix = np.zeros((max_id + 1, embedding_dim), dtype=np.float32)
    type_dict = {}

    for row in node_rows:
        nid = row['id']
        type_dict[nid] = row['node_type']
        embedding_matrix[nid] = np.frombuffer(row['embedding'], dtype=np.float32)

    # Phân loại node
    user_ids = [nid for nid, t in type_dict.items() if t == 'candidate']
    job_ids  = [nid for nid, t in type_dict.items() if t == 'job']
    n_users  = len(user_ids)
    n_jobs   = len(job_ids)
    n_nodes = max_id + 1

    # Tạo embedding riêng
    user_embed   = embedding_matrix[user_ids]
    job_embed    = embedding_matrix[job_ids]
    entity_embed = embedding_matrix  # toàn bộ node embedding

    # 3. Đọc relations và embedding
    cursor.execute("SELECT id, embedding FROM relations")
    rel_rows = cursor.fetchall()

    relation_embed = np.vstack([
        np.frombuffer(r['embedding'], dtype=np.float32)
        for r in rel_rows
    ]).astype(np.float32)

    n_relations = len(rel_rows)

    # 4. Đọc edges
    cursor.execute("SELECT head_node_id, relation_id, tail_node_id FROM edges")
    edge_rows = cursor.fetchall()

    # Tạo ánh xạ ngược từ ID sang index riêng cho CF
    user2cf = {nid: i for i, nid in enumerate(user_ids)}
    job2cf  = {nid: i for i, nid in enumerate(job_ids)}

    # 4a. CF triples: chỉ lấy quan hệ job <-> candidate với relation_id = 3 (HAS_EXPERIENCE)
    cf_h, cf_t = [], []
    for row in edge_rows:
        h_id, t_id = row['head_node_id'], row['tail_node_id']
        h_type, t_type = type_dict[h_id], type_dict[t_id]
        r = row['relation_id']

        if r == 3:
            if h_type == 'job' and t_type == 'candidate':
                cf_h.append(job2cf[h_id])
                cf_t.append(user2cf[t_id])
            elif h_type == 'candidate' and t_type == 'job':
                cf_h.append(job2cf[t_id])
                cf_t.append(user2cf[h_id])

    # Nếu cần tạo ma trận A_cf 
    A_cf = sp.coo_matrix(
        (np.ones(len(cf_h + cf_t), dtype=np.float32), (cf_h + cf_t, cf_t + cf_h)),
        shape=(n_jobs + n_users, n_jobs + n_users)
    )

    # 4b. KG triples: toàn bộ đồ thị
    h_kg = [row['head_node_id'] for row in edge_rows]
    t_kg = [row['tail_node_id'] for row in edge_rows]
    kg_rel = [row['relation_id'] for row in edge_rows]


    rows, cols = h_kg + t_kg, t_kg + h_kg
    A_kg = sp.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_nodes, n_nodes)
    )

    return {
    'config': {
        'n_users': n_users,
        'n_jobs': n_jobs,
        'n_entities': n_nodes,
        'n_relations': n_relations,
        'A_cf': A_cf,
        'A_kg': A_kg,
        'user_embed': user_embed,
        'job_embed': job_embed,
        'entity_embed': entity_embed,
        'relation_embed': relation_embed,
        'cf_triples': (cf_h, [0]*len(cf_h), cf_t),
        'kg_triples': (h_kg, kg_rel, t_kg)
    },
    'job2cf': job2cf,
    'user2cf': user2cf
}



def evaluate_auc(model, sess, test_data, job2cf, user2cf):
    """
    test_data: list các dict {'job': global_job_id, 'candidate': global_user_id, 'label': 0/1}
    """
    y_true, y_score = [], []

    for sample in test_data:
        global_job = sample['job']
        global_user = sample['candidate']
        label = sample['label']

        # Ánh xạ sang chỉ số local
        if global_job not in job2cf or global_user not in user2cf:
            # bỏ qua nếu không tồn tại
            continue
        job_idx = job2cf[global_job]
        user_idx = user2cf[global_user]

        feed_dict = {
            model.jobs: [job_idx],
            model.pos_users: [user_idx],
            model.neg_users: [user_idx],
            model.node_dropout: [0.0] * model.n_layers,
            model.mess_dropout: [0.0] * model.n_layers
        }
        # batch_predictions shape [1,1]
        score = sess.run(model.batch_predictions, feed_dict=feed_dict)[0][0]

        y_true.append(label)
        y_score.append(score)

    if not y_true:
        print("⚠️ Không có sample hợp lệ để đánh giá AUC.")
        return None

    auc = roc_auc_score(y_true, y_score)
    print(f"✅ ROC AUC: {auc:.4f}")
    return auc


def main():
    # 1. Load data và ánh xạ
    data = load_data_from_mysql()  
    data_config = data['config']
    job2cf = data['job2cf']
    user2cf = data['user2cf']

    # 2. Khởi tạo args giống training
    args = Namespace(
        embed_size=384,
        kge_size=384,
        batch_size=256,
        batch_size_kg=128,
        lr=0.001,
        layer_size='[256, 128]',
        alg_type='kgat',
        adj_type='norm',
        adj_uni_type='sum',
        regs='[1e-5, 1e-5]',
        verbose=1
    )

    # 3. Khởi tạo model
    model = KGAT(data_config=data_config, pretrain_data=None, args=args)

    # 4. Load checkpoint
    config = tf.ConfigProto(); config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "saved_model/kgat_model.ckpt")
        print("✅ Model restored.")

        # 5. Load test data
        with open('data_test.json', 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        # 6. Evaluate AUC
        evaluate_auc(model, sess, test_data, job2cf, user2cf)

if __name__ == '__main__':
    main()
