from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import mysql.connector
from KGAT import KGAT
from Main import load_data_from_mysql
from argparse import Namespace

app = Flask(__name__)

# Load model & data
data_config = load_data_from_mysql()
user2cf = data_config['user2cf']
job2cf = data_config['job2cf']
cf2user = {v: k for k, v in user2cf.items()}  # Đảo ngược
cf2job = {v: k for k, v in job2cf.items()}

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
    verbose=0
)

model = KGAT(data_config, None, args)
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, 'saved_model/best_model_fold3.ckpt')
sess.run(tf.global_variables_initializer())

# MySQL Connection
def get_connection():
    return mysql.connector.connect(
        host="localhost", user="root", password="231123", database="cv"
    )

@app.route("/recommend_users", methods=["GET"])
def recommend_users():
    job_id = int(request.args.get("job_id"))
    top_k = int(request.args.get("top_k", 5))

    if job_id not in job2cf:
        return jsonify({"error": "Invalid job_id"}), 400

    job_index = job2cf[job_id]
    user_indices = list(range(len(user2cf)))  # all user indexes

    feed = {
        model.jobs: [job_index],
        model.pos_users: user_indices,
        model.neg_users: user_indices,
        model.node_dropout: [0.0] * model.n_layers,
        model.mess_dropout: [0.0] * model.n_layers
    }
    scores = sess.run(model.batch_predictions, feed_dict=feed)[0]  # shape [n_users]

    # Lấy top-k user theo điểm
    top_k_idx = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_k_idx]

    results = []
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM experience WHERE job_id = %s", (job_id,))
    job_info = cursor.fetchone()
    for i, idx in enumerate(top_k_idx):
        user_id = cf2user[idx]

        # Lấy node_name
        cursor.execute("SELECT node_name FROM nodes WHERE id = %s", (user_id,))
        node = cursor.fetchone()
        node_name = node['node_name'].replace("candidate_", "") if node else "unknown"

        # Lấy thông tin từ bảng candidate
        cursor.execute("SELECT * FROM candidate WHERE candidate_id = %s", (node_name,))
        candidate_info = cursor.fetchone()
        
        # Lấy thông tin từ bảng experience
        cursor.execute("SELECT * FROM cv.experience WHERE candidate_id = %s", (node_name,))
        experiences = cursor.fetchall()

        print("****+++****đây là: ", experiences, node_name)

        results.append({
            "user_id": user_id,
            "score": float(top_scores[i]),
            "node_name": node_name,
            "candidate": candidate_info,
            "experiences": experiences,
            "job_info": job_info
        })

    cursor.close()
    conn.close()
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
