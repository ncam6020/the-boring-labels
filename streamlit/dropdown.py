import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util

# Set the page config
st.set_page_config(layout="centered", page_title="The CSV Boring Machine", page_icon="🐗")

st.title("🐗 The Boring CSV Machine")
st.caption("Replace unique room names with the most similar boring name.")

# Load the model using Streamlit's caching
@st.cache(allow_output_mutation=True)
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# Function to calculate similarities
def calculate_similarities(room_name, boring_names_embeddings):
    input_embedding = model.encode(room_name, convert_to_tensor=True)
    input_embedding = input_embedding.unsqueeze(0)
    similarities = util.pytorch_cos_sim(input_embedding, boring_names_embeddings)
    top_results = torch.topk(similarities, k=len(boring_names_embeddings), dim=1, largest=True, sorted=True)
    return top_results.indices[0].tolist()

# File uploaders in the sidebar
with st.sidebar:
    st.write("## Upload CSV Files")
    uploaded_classifier_file = st.file_uploader("Upload CSV with classifier/boring list", type=["csv"])
    uploaded_room_names_file = st.file_uploader("Upload CSV with original room names", type=["csv"])

if uploaded_classifier_file and uploaded_room_names_file:
    classifier_df = pd.read_csv(uploaded_classifier_file)
    room_names_df = pd.read_csv(uploaded_room_names_file)

    # Aggregate and process room names
    aggregated_df = room_names_df.groupby('Original Room Name').size().reset_index(name='Unique Count')
    boring_names = classifier_df['Boring Name'].tolist()
    boring_names_embeddings = torch.stack([model.encode(name, convert_to_tensor=True) for name in boring_names])

    # Prepare data for annotation with the top match as the default
    annotations_data = []
    for _, row in aggregated_df.iterrows():
        original_name = row['Original Room Name']
        unique_count = row['Unique Count']
        top_match_indices = calculate_similarities(original_name, boring_names_embeddings)
        top_matches = [boring_names[i] for i in top_match_indices[:3]]
        all_options = sorted(boring_names)  # Sort all boring names alphabetically

        annotations_data.append({
            "Original Room Name": original_name,
            "Unique Count": unique_count,
            "Selected Boring Name": top_matches[0],  # Default to top match
            "Boring Name Options": all_options
        })

    df = pd.DataFrame(annotations_data)
    df['Selected Boring Name'] = pd.Categorical(df['Selected Boring Name'], categories=boring_names)

    # Use st.data_editor with SelectboxColumn
    st.data_editor(
        df,
        column_config={
            "Selected Boring Name": st.column_config.SelectboxColumn(
                label="Choose Boring Name",
                width="medium",
                options=boring_names,
                default=df['Selected Boring Name'].cat.categories[0]  # Default to first category
            )
        },
        hide_index=True
    )
    
    user_selections_df = pd.DataFrame(user_selections_data)

    if st.button('Export to CSV'):
        export_df = room_names_df.merge(user_selections_df, on='Original Room Name', how='left')
        export_df.rename(columns={'User Selection': 'Selected Boring Name', 'Top Suggestion': 'Suggested Boring Name'}, inplace=True)
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download updated names CSV", data=csv, file_name="updated_names.csv", mime="text/csv")
