##############################env\Scripts\Activate.ps1#######################
import pandas as pd
import streamlit as st
import torch
import os
import json
from sentence_transformers import SentenceTransformer, util

# Get the directory of the current script
base_dir = os.path.dirname(__file__)


# Set Streamlit page configuration
st.set_page_config(layout="centered", page_title="The CSV Boring Machine", page_icon="🐗")

# Streamlit container
with st.container():
    st.title("🐗 The Boring CSV Machine")
    st.caption("Replace unique room names with the most similar boring name.")

# Function to load the SentenceTransformer model
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
    return top_results.indices[0].tolist(), top_results.values[0].tolist()

# Streamlit sidebar for file upload
with st.sidebar:
    st.write("## Upload CSV Files")
    uploaded_classifier_file = st.file_uploader("Upload CSV with classifier/boring list", type=["csv"])
    uploaded_room_names_file = st.file_uploader("Upload CSV with original room names", type=["csv"])

# Initialize an empty DataFrame for final_csv_df
final_csv_df = pd.DataFrame()

# Check if files are uploaded
if uploaded_classifier_file is not None and uploaded_room_names_file is not None:
    classifier_df = pd.read_csv(uploaded_classifier_file)
    room_names_df = pd.read_csv(uploaded_room_names_file)

    # Group and aggregate data
    aggregated_df = room_names_df.groupby('Original Room Name').size().reset_index(name='Unique Count')
    boring_names = classifier_df['Boring Name'].tolist()
    boring_names_embeddings = torch.stack([model.encode(name, convert_to_tensor=True) for name in boring_names])

    # Prepare data for annotations
    annotations_data = []
    for _, row in aggregated_df.iterrows():
        original_name = row['Original Room Name']
        unique_count = row['Unique Count']
        top_match_indices, top_match_scores = calculate_similarities(original_name, boring_names_embeddings)
        
        # Append data with the top match score
        annotations_data.append({
            "Original Room Name": original_name,
            "Unique Count": unique_count,
            "Selected Boring Name": f"{boring_names[top_match_indices[0]]} -{top_match_scores[0]*100:.0f}%",
            "Top Boring Name Suggestion": boring_names[top_match_indices[0]],
            "Top Match Score": top_match_scores[0],
            "Boring Name Options": sorted(boring_names)
        })

    # Create DataFrame from annotations
    df = pd.DataFrame(annotations_data)
    df_sorted = df.sort_values(by=['Unique Count', 'Original Room Name'], ascending=[False, True])
    df_for_editor = df_sorted[['Unique Count', 'Original Room Name', 'Selected Boring Name']]

    # Streamlit container for data editor
    with st.container():
        edited_df = st.data_editor(
            df_for_editor,
            column_config={
                "Selected Boring Name": st.column_config.SelectboxColumn(
                    options=df['Boring Name Options'].iloc[0],
                    default=df['Selected Boring Name'].iloc[0]
                )
            },
            hide_index=True,
            use_container_width=True
        )

        # Update DataFrame based on edited data
        if edited_df is not None:
            df['Selected Boring Name'] = edited_df['Selected Boring Name']

        # Merge and create final DataFrame
        final_df = room_names_df.merge(
            df[['Original Room Name', 'Selected Boring Name', 'Top Boring Name Suggestion', 'Top Match Score']],
            on='Original Room Name',
            how='left'
        )

        # Removing probability score from 'Selected Boring Name' in final CSV
        final_df['Selected Boring Name'] = final_df['Selected Boring Name'].str.split(' -').str[0]

        # Update final_csv_df
        final_csv_df = final_df[['Original Room Name', 'Selected Boring Name', 'Top Boring Name Suggestion', 'Top Match Score']]

        # Streamlit download button and appending logic
        downloaded = st.download_button(
            "⬇️ Download Your Boring Name File",
            final_csv_df.to_csv(index=False),
            "annotated_room_names.csv",
            mime='text/csv',
            use_container_width=True
        )

        # File path for the existing JSON file
        json_file_path = os.path.join(base_dir, "trainingdata", "training.json")

        # Append data when download button is clicked
        if downloaded:
            if not final_csv_df.empty:
                try:
                    # Convert DataFrame to JSON
                    data_json = final_csv_df.to_json(orient='records')

                    # Check if JSON file exists
                    if os.path.exists(json_file_path):
                        with open(json_file_path, 'r+') as file:
                            # Read existing data into a list
                            file_data = json.load(file)
                            # Check if file_data is a list and append new data
                            if isinstance(file_data, list):
                                file_data.extend(json.loads(data_json))
                                # Set file's current position at offset
                                file.seek(0)
                                file.truncate()
                                # Write updated data
                                json.dump(file_data, file, indent=4)
                            else:
                                raise ValueError("JSON file does not contain a list")
                    else:
                        # Create new JSON file and write data
                        with open(json_file_path, 'w') as file:
                            json.dump(json.loads(data_json), file, indent=4)
                    
                    st.success("You Have Now Contributed to the Boring AI Model -thanks!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("The DataFrame to be appended is empty.")

