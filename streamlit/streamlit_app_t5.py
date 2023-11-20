import pandas as pd
import streamlit as st
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Set the page config
st.set_page_config(layout="centered", page_title="The Boring Name Annotator", page_icon="🧮")

st.title("✍️ Boring Name Annotator")
st.caption("Annotate each unique room name with the most similar boring name.")

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

# File uploaders for CSV files
uploaded_classifier_file = st.file_uploader("Upload CSV with classifier/boring list", type=["csv"])
uploaded_room_names_file = st.file_uploader("Upload CSV with original room names", type=["csv"])

if uploaded_classifier_file and uploaded_room_names_file:
    classifier_df = pd.read_csv(uploaded_classifier_file)
    room_names_df = pd.read_csv(uploaded_room_names_file)

    # Aggregate and process room names
    aggregated_df = room_names_df.groupby('Original Room Name').size().reset_index(name='Unique Count')
    boring_names = classifier_df['Boring Name'].tolist()
    boring_names_embeddings = torch.stack([model.encode(name, convert_to_tensor=True) for name in boring_names])

    # Prepare data for annotation
    annotations_data = []
    for _, row in aggregated_df.iterrows():
        original_name = row['Original Room Name']
        unique_count = row['Unique Count']
        top_match_indices = calculate_similarities(original_name, boring_names_embeddings)
        top_matches = [boring_names[i] for i in top_match_indices[:3]]
        annotations_data.append({
            "Original Name": original_name,
            "Unique Count": unique_count,
            "Top Boring Name Suggestion": top_matches[0]
        })

    df = pd.DataFrame(annotations_data)

    # Combine current and new categories, ensuring no duplicates
    all_possible_categories = set(boring_names).union(df['Top Boring Name Suggestion'])

    # Set the combined list as the new categories
    df['Top Boring Name Suggestion'] = pd.Categorical(df['Top Boring Name Suggestion'], categories=all_possible_categories)

    annotated = st.data_editor(df, hide_index=True, use_container_width=True, disabled=["Original Name", "Unique Count"])

    st.download_button(
        "⬇️ Download annotations as .csv",
        annotated.to_csv(index=False),
        "annotated_data.csv",
        use_container_width=True
    )
