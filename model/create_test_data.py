import mysql.connector
import json
import csv
import random

# ====== 1. Kết nối tới MySQL ======
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='231123',
    database='cv'  # Thay bằng tên database thực tế
)

cursor = conn.cursor(dictionary=True)

# ====== 2. Đọc các node thuộc loại job (position) và candidate ======
cursor.execute("SELECT id, node_type FROM nodes WHERE node_type IN ('job', 'candidate') ")
nodes = cursor.fetchall()

jobs = [node['id'] for node in nodes if node['node_type'] == 'job']
candidates = [node['id'] for node in nodes if node['node_type'] == 'candidate']


# ====== 3. Đọc bảng edges để lấy quan hệ ======
cursor.execute("SELECT head_node_id, tail_node_id, relation_id FROM edges")
edges = cursor.fetchall()

# ====== 4. Tạo positive pairs (job, candidate) có kết nối ======
positive_pairs = []
for edge in edges:
    head = edge['head_node_id']
    tail = edge['tail_node_id']
    if head in candidates and tail in jobs:
        positive_pairs.append((tail, head))  # job, candidate
    elif head in jobs and tail in candidates:
        positive_pairs.append((head, tail))  # job, candidate

# Bỏ trùng nếu có
positive_pairs = list(set(positive_pairs))

# ====== 5. Tạo negative pairs (các cặp không kết nối) ======
negative_pairs = set()
while len(negative_pairs) < len(positive_pairs):
    job = random.choice(jobs)
    candidate = random.choice(candidates)
    if (job, candidate) not in positive_pairs:
        negative_pairs.add((job, candidate))

# ====== 6. Tạo data_test ======
data_test = []

# Thêm positive
for job, candidate in positive_pairs:
    data_test.append({'job': job, 'candidate': candidate, 'label': 1})

# Thêm negative
for job, candidate in negative_pairs:
    data_test.append({'job': job, 'candidate': candidate, 'label': 0})

# ====== 7. Lưu ra file JSON ======
with open('data_test.json', 'w', encoding='utf-8') as f_json:
    json.dump(data_test, f_json, ensure_ascii=False, indent=4)



# ====== 9. Đóng kết nối ======
cursor.close()
conn.close()
