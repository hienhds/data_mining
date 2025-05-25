# recommender.py
import numpy as np

def recommend_top_k_users(job_id, model, sess, data_config, k=5):
    user_ids = list(range(data_config['n_users']))
    feed_dict = {
        model.jobs: [job_id] * len(user_ids),
        model.pos_users: user_ids,
        model.neg_users: user_ids,
        model.node_dropout: [0.0] * model.n_layers,
        model.mess_dropout: [0.0] * model.n_layers
    }
    scores_matrix = model.eval(sess, feed_dict)
    scores = scores_matrix[0]
    top_k_indices = np.argsort(scores)[::-1][:k]
    return top_k_indices.tolist()
