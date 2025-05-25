from flask import Flask, request, jsonify
from recommender import recommend_top_k_users
from model_loader import load_model

app = Flask(__name__)
model, sess, data_config = load_model()

@app.route("/recommend", methods=["GET"])
def recommend():
    job_id = int(request.args.get("job_id", 0))
    top_users = recommend_top_k_users(job_id, model, sess, data_config)
    return jsonify({"job_id": job_id, "top_candidates": top_users})

if __name__ == "__main__":
    app.run(debug=True)