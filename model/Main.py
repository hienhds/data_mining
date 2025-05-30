import mysql.connector
import scipy.sparse as sp
import tensorflow as tf
import numpy as np
from KGAT import KGAT
from argparse import Namespace
import random
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from collections import defaultdict
import heapq

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
        'kg_triples': (h_kg, kg_rel, t_kg),
        'user2cf': user2cf,
        'job2cf': job2cf
    }

def evaluate_metrics(model, sess, job2cf, user2cf, test_pairs):
    y_true, y_pred, y_score = [], [], []
    for pair in test_pairs:
        job, user, label = pair['job'], pair['user'], pair['label']
        if job not in job2cf or user not in user2cf:
            continue
        feed = {
            model.jobs: [job2cf[job]],
            model.pos_users: [user2cf[user]],
            model.neg_users: [user2cf[user]],
            model.node_dropout: [0.0] * model.n_layers,
            model.mess_dropout: [0.0] * model.n_layers
        }
        score = sess.run(model.batch_predictions, feed_dict=feed)[0][0]
        y_score.append(score)
        y_pred.append(1 if score >= 0.5 else 0)
        y_true.append(label)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_score)
    }

from collections import defaultdict
import heapq

def recall_at_k(ranked_list, ground_truth, k):
    return len(set(ranked_list[:k]) & set(ground_truth)) / float(len(ground_truth))

def ndcg_at_k(ranked_list, ground_truth, k):
    dcg = 0.0
    for i in range(k):
        if ranked_list[i] in ground_truth:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_ranking(model, sess, job2cf, user2cf, test_pairs, K_list=[5,10,20]):
    # Xây index các job từng apply bởi mỗi user
    user_positive = defaultdict(set)
    for pair in test_pairs:
        if pair['label'] == 1:
            user_positive[pair['user']].add(pair['job'])

    all_jobs = list(job2cf.keys())
    recall_k_result = {k: [] for k in K_list}
    ndcg_k_result = {k: [] for k in K_list}

    for user in user_positive:
        if user not in user2cf:
            continue
        gt_jobs = list(user_positive[user])
        candidate_jobs = [j for j in all_jobs if j not in gt_jobs]

        # Gộp GT + negative sample
        jobs_to_score = candidate_jobs + gt_jobs
        feed = {
            model.jobs: [job2cf[j] for j in jobs_to_score],
            model.pos_users: [user2cf[user]] * len(jobs_to_score),
            model.neg_users: [user2cf[user]] * len(jobs_to_score),
            model.node_dropout: [0.0] * model.n_layers,
            model.mess_dropout: [0.0] * model.n_layers
        }
        scores = sess.run(model.batch_predictions, feed_dict=feed)[0]
        ranked_jobs = [jobs_to_score[i] for i in np.argsort(scores)[::-1]]

        for k in K_list:
            recall = recall_at_k(ranked_jobs, gt_jobs, k)
            ndcg = ndcg_at_k(ranked_jobs, gt_jobs, k)
            recall_k_result[k].append(recall)
            ndcg_k_result[k].append(ndcg)

    results = {}
    for k in K_list:
        results[f"Recall@{k}"] = np.mean(recall_k_result[k])
        results[f"nDCG@{k}"] = np.mean(ndcg_k_result[k])
    return results


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
    # Lấy KG triples
    h_list, r_list, t_list = data_config['kg_triples']
    # Tổng số node trong KG = jobs + entities
    n_nodes = data_config['n_jobs'] + data_config['n_entities']

    # Ghép thành list triple, shuffle
    triples = list(zip(h_list, r_list, t_list))
    random.shuffle(triples)

    batches = []
    for i in range(0, len(triples), batch_size):
        batch = triples[i:i + batch_size]
        h_b, r_b, pos_t_b = zip(*batch)
        # Neg sampling: random từ toàn bộ node space
        neg_t_b = random.choices(list(range(n_nodes)), k=len(batch))
        batches.append({
            'h':     list(h_b),
            'r':     list(r_b),
            'pos_t': list(pos_t_b),
            'neg_t': neg_t_b
        })
    return batches

def get_all_job_user_pairs(user2cf, job2cf):
    conn = mysql.connector.connect(
        host="localhost", user="root", password="231123", database="cv"
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT head_node_id, relation_id, tail_node_id FROM edges")
    rows = cursor.fetchall()
    conn.close()

    pairs = []
    for row in rows:
        h, r, t = row['head_node_id'], row['relation_id'], row['tail_node_id']
        if r == 3:
            if h in job2cf and t in user2cf:
                pairs.append({'job': h, 'user': t, 'label': 1})
            if t in job2cf and h in user2cf:
                pairs.append({'job': t, 'user': h, 'label': 1})

    # Sinh negative sample
    all_jobs = list(job2cf.keys())
    all_users = list(user2cf.keys())
    for _ in range(len(pairs)):
        j = random.choice(all_jobs)
        u = random.choice(all_users)
        if {'job': j, 'user': u, 'label': 1} not in pairs:
            pairs.append({'job': j, 'user': u, 'label': 0})

    return pairs



def main():
    # Load dữ liệu
    data_config = load_data_from_mysql()

    args = Namespace(
        embed_size=384,
        kge_size=384,
        batch_size=512,
        batch_size_kg=1024,
        lr=0.0001,
        layer_size='[256, 128]',
        alg_type='kgat',
        adj_type='norm',
        adj_uni_type='sum',
        regs='[1e-5, 1e-5]',
        verbose=1
    )

    user2cf = data_config['user2cf']
    job2cf = data_config['job2cf']

    all_pairs = get_all_job_user_pairs(user2cf, job2cf)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_auc = 0
    best_model_path = ""

    fold_idx = 1
    for train_idx, test_idx in kf.split(all_pairs):
        print(f"\n===== Fold {fold_idx} =====")
        fold_idx += 1

        train_pairs = [all_pairs[i] for i in train_idx]
        test_pairs = [all_pairs[i] for i in test_idx]

        model = KGAT(data_config=data_config, pretrain_data=None, args=args)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in tqdm(range(1, 101), desc="Training Epochs"):
                for cf_batch in generate_cf_batches(data_config, args.batch_size):
                    feed_dict = {
                        model.jobs: cf_batch['job_ids'],
                        model.pos_users: cf_batch['pos_user_ids'],
                        model.neg_users: cf_batch['neg_user_ids'],
                        model.node_dropout: [0.1] * len(eval(args.layer_size)),
                        model.mess_dropout: [0.1] * len(eval(args.layer_size))
                    }
                    model.train(sess, feed_dict)

                for kg_batch in generate_kg_batches(data_config, args.batch_size_kg):
                    feed_dict = {
                        model.h: kg_batch['h'],
                        model.r: kg_batch['r'],
                        model.pos_t: kg_batch['pos_t'],
                        model.neg_t: kg_batch['neg_t']
                    }
                    model.train_A(sess, feed_dict)

                model.update_attentive_A(sess)

            # 🎯 Đánh giá sau mỗi fold
            metrics = evaluate_metrics(model, sess, job2cf, user2cf, test_pairs)
            print("📊 Evaluation:", metrics)
            ranking_metrics = evaluate_ranking(model, sess, job2cf, user2cf, test_pairs)
            print("📈 Ranking Evaluation:", ranking_metrics)

            
            saver = tf.train.Saver()
            best_model_path = f"saved_model/best_model_fold{fold_idx-1}.ckpt"
            saver.save(sess, best_model_path)
            print(f"🌟 Saved model")



if __name__ == "__main__":
    main()
    # data_config = load_data_from_mysql()
    # print(data_config['A_cf'])
    # count_ones = np.sum(data_config['A_cf'].data == 1.0)
    # print(f"Số lượng cạnh có giá trị 1.0 trong A_cf: {count_ones}")

