import pandas as pd
import mysql.connector

# Kết nối đến MySQL
conn = mysql.connector.connect(
    host='localhost',           # hoặc IP database server
    user='root',       # tên đăng nhập MySQL
    password='231123',   # mật khẩu
    database='cv'               # tên database
)

# Đọc dữ liệu từ bảng experience
df_experience = pd.read_sql("SELECT * FROM experience", conn)

# Đọc dữ liệu từ bảng candidate
df_candidate = pd.read_sql("SELECT * FROM candidate", conn)

# Ghi ra file Excel
df_experience.to_excel("experience.xlsx", index=False)
df_candidate.to_excel("candidate.xlsx", index=False)

# Đóng kết nối
conn.close()

print("Xuất dữ liệu thành công ra 'experience.xlsx' và 'candidate.xlsx'")
