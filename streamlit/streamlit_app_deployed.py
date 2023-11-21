##############################env\Scripts\Activate.ps1#######################
import pandas as pd
import streamlit as st
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Set the page config
st.set_page_config(layout="centered", page_title="The CSV Boring Machine", page_icon="🐗")

st.title("🐗 The Boring CSV Machine")
st.caption("Replace unique room names with the most similar boring name.")

# Load the model using Streamlit's caching
@st.cache_data
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

    # Prepare data for annotation
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
            "Top Boring Name Suggestion": top_matches[0],
            "Boring Name Options": all_options
        })

    df = pd.DataFrame(annotations_data)

    # Set categories for the 'Selected Boring Name' column
    categories = df['Boring Name Options'][0]
    df['Selected Boring Name'] = pd.Categorical(df['Selected Boring Name'], categories=categories)

    # Sort the DataFrame by 'Unique Count' (descending) and then by 'Original Room Name' (alphabetically)
    df_sorted = df.sort_values(by=['Unique Count', 'Original Room Name'], ascending=[False, True])

    # Rearrange columns to put 'Unique Count' on the left
    df_display = df_sorted[['Unique Count', 'Original Room Name', 'Selected Boring Name']].copy()

    # Display DataFrame in data editor
    annotated = st.data_editor(df_display, hide_index=True, use_container_width=True)

   # Prepare the final DataFrame for CSV download
    if annotated is not None:
        final_df = room_names_df.merge(
            annotated[['Original Room Name', 'Selected Boring Name']],
            on='Original Room Name',
            how='left'
        ).merge(
            df[['Original Room Name', 'Top Boring Name Suggestion']],
            on='Original Room Name',
            how='left'
        )

        # Exclude 'Unique Count' from the final CSV
        final_csv_df = final_df[['Original Room Name', 'Selected Boring Name', 'Top Boring Name Suggestion']]

        # Download button for the final CSV
        st.download_button(
            "⬇️ Download annotations as .csv",
            final_csv_df.to_csv(index=False),
            "annotated_room_names.csv",
            mime='text/csv',
            use_container_width=True
        )
