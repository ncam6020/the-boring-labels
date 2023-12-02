##############################env\Scripts\Activate.ps1#######################
import pandas as pd
import streamlit as st
import torch
import os
import json
from sentence_transformers import SentenceTransformer, util

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
    top_results = torch.topk(similarities, k=3, dim=1, largest=True, sorted=True)
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
    boring_name_options = sorted(classifier_df['Boring Name'].tolist())
    for _, row in aggregated_df.iterrows():
        original_name = row['Original Room Name']
        unique_count = row['Unique Count']
        top_match_indices, top_match_scores = calculate_similarities(original_name, boring_names_embeddings)

        annotations_data.append({
            "Original Room Name": original_name,
            "Unique Count": unique_count,
            "Selected Boring Name": f"{boring_names[top_match_indices[0]]} -{top_match_scores[0]*100:.0f}%",
            "2nd Boring Name Suggestion": f"{boring_names[top_match_indices[1]]} -{top_match_scores[1]*100:.0f}%",
            "3rd Boring Name Suggestion": f"{boring_names[top_match_indices[2]]} -{top_match_scores[2]*100:.0f}%",
            "Top Match Score": top_match_scores[0]
        })

    df = pd.DataFrame(annotations_data)
    df_sorted = df.sort_values(by=['Unique Count', 'Original Room Name'], ascending=[False, True])
    df_for_editor = df_sorted[['Unique Count', 'Original Room Name', 'Selected Boring Name']]

    with st.container():
        edited_df = st.data_editor(
            df_for_editor,
            column_config={
                "Selected Boring Name": st.column_config.SelectboxColumn(
                    options=boring_name_options,
                    default=df_for_editor['Selected Boring Name'].iloc[0] if not df_for_editor.empty else None
                )
            },
            hide_index=True,
            use_container_width=True
        )

        if edited_df is not None:
            df.update(edited_df, overwrite=True)

        df['Top Boring Name Suggestion'] = df['Selected Boring Name'].str.split(' -').str[0]

        final_df = room_names_df.merge(
            df[['Original Room Name', 'Selected Boring Name', 'Top Boring Name Suggestion', '2nd Boring Name Suggestion', '3rd Boring Name Suggestion', 'Top Match Score']],
            on='Original Room Name',
            how='left'
        )

        final_csv_df = final_df.copy()
        final_csv_df['Selected Boring Name'] = final_csv_df['Selected Boring Name'].str.split(' -').str[0]
        final_csv_df['2nd Boring Name Suggestion'] = final_csv_df['2nd Boring Name Suggestion'].str.split(' -').str[0]
        final_csv_df['3rd Boring Name Suggestion'] = final_csv_df['3rd Boring Name Suggestion'].str.split(' -').str[0]

        # Streamlit download button for final CSV
        if st.download_button(
            "⬇️ Download Your Boring Name File",
            final_csv_df.to_csv(index=False),
            "annotated_room_names.csv",
            mime='text/csv',
            use_container_width=True
        ):
            if not final_csv_df.empty:
                try:
                    # Append data to the JSON file
                    data_to_append = final_csv_df[['Original Room Name', 'Selected Boring Name', '2nd Boring Name Suggestion', '3rd Boring Name Suggestion', 'Top Match Score']]
                    final_json_data = data_to_append.to_json(orient='records')
                    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trainingdata", "training.json")

                    if os.path.exists(json_file_path):
                        with open(json_file_path, 'r+') as file:
                            file_data = json.load(file)
                            if isinstance(file_data, list):
                                file_data.extend(json.loads(final_json_data))
                                file.seek(0)
                                file.truncate()
                                json.dump(file_data, file, indent=4)
                            else:
                                raise ValueError("JSON file does not contain a list")
                    else:
                        with open(json_file_path, 'w') as file:
                            json.dump(json.loads(final_json_data), file, indent=4)

                    st.success("You Have Now Contributed to the Boring AI Model - thanks!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("The DataFrame to be appended is empty.")










