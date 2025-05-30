import streamlit as st
import requests

st.set_page_config(page_title="KGAT Job Matching", layout="wide")
st.title("🔍 Đề xuất ứng viên phù hợp theo Job ID (KGAT)")

job_id = st.number_input("Nhập Job ID", min_value=0)
top_k = st.slider("Số ứng viên muốn gợi ý", min_value=1, max_value=20, value=5)

if st.button("Tìm ứng viên"):
    with st.spinner("Đang truy vấn mô hình..."):
        try:
            res = requests.get("http://localhost:5000/recommend_users", params={
                "job_id": int(job_id),
                "top_k": int(top_k)
            })
            if res.status_code != 200:
                st.error(res.json().get("error", "Lỗi không xác định"))
            else:
                users = res.json()
                job_info = users[0].get("job_info")
                if job_info:
                    st.subheader("🧾 Thông tin công việc")
                    st.write(f"**Tiêu đề công việc:** {job_info.get('title', '')}")
                    st.write(f"**Các kỹ năng yêu cầu:** {job_info.get('skill', '')}")
                    st.write(f"**Mô tả:** {job_info.get('description', '')}")

                else:
                    st.warning("Không tìm thấy thông tin công việc.")

                st.success(f"Tìm được {len(users)} ứng viên phù hợp!")

                for user in users:
                    st.markdown("----")
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.subheader(f"🎯 {user['node_name']}")
                        st.write(f"**Score:** {round(user['score'], 4)}")
                        if user["candidate"]:
                            st.write("**Headline:**", user["candidate"].get("title", ""))
                            st.write("**Position Curent:**", user["candidate"].get("position_title", ""))
                            st.write("**Domain Area:**", user["candidate"].get("domain_area", ""))
                            st.write("**Education:**", user["candidate"].get("education", ""))
                            st.write("**Skill:**", user["candidate"].get("skill", ""))
                            st.write("**Language:**", user["candidate"].get("language", ""))
                            st.write("**Summary:**", user["candidate"].get("summary", ""))
                        
                        else:
                            st.warning("Không có thông tin trong bảng `candidate`")

                    with col2:
                        st.subheader("📌 Kinh nghiệm")
                        exp_list = user.get("experiences", [])
                        print(exp_list)
                        print(user['experiences'], 1)
                        if exp_list:
                            for exp in exp_list:
                                if exp.get('job_id') != job_info.get('job_id'):
                                    st.markdown(f"""
                                    - **Vị trí:** {exp.get('job_role', 'Không rõ')}
                                    - **Thời gian:** {exp.get('years_experience', '')} 
                                    - **Mô tả:** {exp.get('description', '')}
                                    """)
                        else:
                            st.info("Chưa có kinh nghiệm nào được ghi nhận.")

        except Exception as e:
            st.error(f"Lỗi khi gọi API: {e}")
