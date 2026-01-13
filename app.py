# ====================================================
# app.py - 3D Interactive AI Career Recommendation
# ====================================================

import streamlit as st
import torch
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# --------------------------
# 1️⃣ Paths
# --------------------------
data_path = r"C:\Users\MASTER COMPUTERS\Desktop\websitetask\facebook\Documents\Downloads\archive"

model_path = f"{data_path}\\career_model.pth"
vectorizer_path = f"{data_path}\\tfidf_vectorizer.pkl"
embeddings_path = f"{data_path}\\career_embeddings.pkl"
postings_csv = f"{data_path}\\linkedin_job_postings.csv"
summaries_csv = f"{data_path}\\job_summary.csv"
skills_csv = f"{data_path}\\job_skills.csv"

# --------------------------
# 2️⃣ Load Dataset & Model
# --------------------------
postings = pd.read_csv(postings_csv)
summaries = pd.read_csv(summaries_csv)
skills_df = pd.read_csv(skills_csv)

df = postings.merge(summaries, on="job_link", how="inner").merge(skills_df, on="job_link", how="inner")
df['text'] = df['job_summary'].fillna('') + " " + df['job_skills'].fillna('')

with open(embeddings_path, "rb") as f:
    career_embeddings = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

class CareerNN(torch.nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super(CareerNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 512)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(512, embedding_dim)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CareerNN(input_dim=5000, embedding_dim=128)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# --------------------------
# 3️⃣ Streamlit UI
# --------------------------
st.set_page_config(page_title="3D Career Dashboard", page_icon="🚀", layout="wide")
st.title("🚀 3D Interactive AI Career Recommendation Dashboard")
st.write("Visualize your skill coverage and get career recommendations!")

# Sidebar filters
st.sidebar.header("Filters")
top_k = st.sidebar.slider("Number of Recommendations", 1, 20, 5)
job_type_filter = st.sidebar.multiselect("Job Type", df['job_type'].dropna().unique()) if 'job_type' in df.columns else []
remote_filter = st.sidebar.selectbox("Remote Option", ["Any", "Remote Only", "Onsite Only"]) if 'remote' in df.columns else "Any"

user_input = st.text_area("Enter your skills, experience, and interests (comma separated):")

# --------------------------
# 4️⃣ Recommendation Function
# --------------------------
def recommend_jobs(user_input, model, vectorizer, career_embeddings, df, top_k=5):
    if not user_input.strip():
        return pd.DataFrame()
    
    user_vec = vectorizer.transform([user_input]).toarray()
    user_tensor = torch.tensor(user_vec, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        user_emb = model(user_tensor).cpu().numpy()
    
    sims = cosine_similarity(user_emb, career_embeddings)[0]
    df_copy = df.copy()
    df_copy['score'] = sims
    
    if job_type_filter:
        df_copy = df_copy[df_copy['job_type'].isin(job_type_filter)]
    if 'remote' in df_copy.columns and remote_filter != "Any":
        df_copy = df_copy[df_copy['remote'] == ("Yes" if remote_filter=="Remote Only" else "No")]
    
    return df_copy.sort_values(by='score', ascending=False).head(top_k)

# --------------------------
# 5️⃣ Run Recommendations
# --------------------------
if st.button("Get Recommendations"):
    if not user_input.strip():
        st.warning("Please enter your skills!")
    else:
        results = recommend_jobs(user_input, model, vectorizer, career_embeddings, df, top_k=top_k)
        if results.empty:
            st.warning("No recommendations found for your filters!")
        else:
            st.success(f"Top {top_k} Recommendations")
            
            # --------------------------
            # 3D Skill Coverage Scatter
            # --------------------------
            user_skills_set = set([s.strip().lower() for s in user_input.split(",")])
            scatter_data = []

            for idx, row in results.iterrows():
                job_skills = set([s.strip().lower() for s in row['job_skills'].split(",")]) if pd.notna(row['job_skills']) else set()
                matched = len(user_skills_set.intersection(job_skills))
                missing = len(job_skills - user_skills_set)
                total = len(job_skills) if job_skills else 1
                
                scatter_data.append({
                    'Job': row['job_title'],
                    'Matched': matched,
                    'Missing': missing,
                    'Total': total,
                    'Score': row['score']
                })

            scatter_df = pd.DataFrame(scatter_data)
            fig = px.scatter_3d(scatter_df,
                                x='Matched',
                                y='Missing',
                                z='Score',
                                color='Job',
                                size='Total',
                                hover_name='Job',
                                size_max=30,
                                title="3D Skill Coverage of Top Recommendations")
            st.plotly_chart(fig, use_container_width=True)
            
            # --------------------------
            # Detailed Recommendations with Progress
            # --------------------------
            for idx, row in results.iterrows():
                st.subheader(f"{row['job_title']} at {row['company']} ({row['score']:.2f})")
                st.write(f"**Job Description:** {row['job_summary']}")
                st.write(f"**Skills Required:** {row['job_skills']}")
                
                job_skills = set([s.strip().lower() for s in row['job_skills'].split(",")]) if pd.notna(row['job_skills']) else set()
                matched = len(user_skills_set.intersection(job_skills))
                total = len(job_skills) if job_skills else 1
                st.progress(int((matched/total)*100))
                st.markdown(f"Matched Skills: {matched}/{total}")
                st.markdown("---")
            
            # --------------------------
            # Top Skills Chart
            # --------------------------
            top_skills = []
            for sk in results['job_skills'].dropna():
                top_skills.extend([s.strip() for s in sk.split(",")])
            skill_counts = pd.Series(top_skills).value_counts().head(20)
            fig2 = px.bar(skill_counts, x=skill_counts.values, y=skill_counts.index,
                          orientation='h', text=skill_counts.values,
                          title="Top Skills in Recommended Jobs")
            st.plotly_chart(fig2, use_container_width=True)
            
            # --------------------------
            # Export to CSV
            # --------------------------
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Recommendations as CSV",
                data=csv,
                file_name='career_recommendations.csv',
                mime='text/csv'
            )
