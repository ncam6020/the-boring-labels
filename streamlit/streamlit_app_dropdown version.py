import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# Set the page config at the start of the script
st.set_page_config(layout="centered", page_title="The Boring CSV Machine", page_icon="🐗")

# Load the model using Streamlit's caching
def load_model():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model

model = load_model()

# Function to calculate similarities
def calculate_similarities(room_name, boring_names_embeddings):
    input_embedding = model.encode(room_name, convert_to_tensor=True)
    input_embedding = input_embedding.unsqueeze(0)  # Ensure it's a 2D tensor
    similarities = util.pytorch_cos_sim(input_embedding, boring_names_embeddings)
    top_results = torch.topk(similarities, k=3, dim=1, largest=True, sorted=True)
    return top_results.indices[0].tolist(), top_results.values[0].tolist()

# Sidebar for API key
API_KEY = st.sidebar.text_input("Enter your HuggingFace API key", type="password")

# Main content
st.title("The Boring CSV Machine")

# File uploaders for both CSV files
uploaded_classifier_file = st.file_uploader("Upload CSV with classifier/boring list", type=["csv"], key="classifier_uploader")
uploaded_room_names_file = st.file_uploader("Upload CSV with original room names", type=["csv"], key="room_names_uploader")

# Initialize display DataFrame
display_df = pd.DataFrame(columns=['Original Name', 'Unique Count', 'Boring Name'])

# When both files are uploaded
if uploaded_classifier_file and uploaded_room_names_file:
    classifier_df = pd.read_csv(uploaded_classifier_file)
    room_names_df = pd.read_csv(uploaded_room_names_file)

    boring_names = classifier_df['Boring Name'].tolist()
    boring_names_embeddings = torch.stack([model.encode(name, convert_to_tensor=True) for name in boring_names])

    # Create a form and display the table with the selection boxes
    user_selections_data = []
    initial_suggestion = []
    index = -1
    
    # with st.form(key='selection_form'):
    for original_name in room_names_df['Original Room Name'].unique():
        best_match_indices, best_match_scores = calculate_similarities(original_name, boring_names_embeddings)
        top_matches = [boring_names[i] for i in best_match_indices]
        remaining_boring_names = sorted(set(boring_names) - set(top_matches))
        dropdown_options = [f"{top_matches[1]} ({best_match_scores[1]:.0%})", 
                            f"{top_matches[2]} ({best_match_scores[2]:.0%})"] + remaining_boring_names
        top_suggestion = f"{top_matches[0]} ({best_match_scores[0]:.0%})"
        initial_suggestion.append(top_matches[0])
        if top_suggestion not in dropdown_options:
            dropdown_options.insert(0, top_suggestion)

        
        # Display the row with three columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.text(original_name)
        with col2:
            st.text(str(room_names_df[room_names_df['Original Room Name'] == original_name].shape[0]))
        with col3:
            user_selection = st.selectbox(
                "",
                options=dropdown_options,
                index=dropdown_options.index(top_suggestion),
                key=f"select_{original_name}"
            )
        index = index + 1
        user_selections_data.append({
        'Original Room Name': original_name, 
        'User Selection': user_selection.split(' (')[0],
        'Top Suggestion': initial_suggestion[index]
    })
    
    
    user_selections_df = pd.DataFrame(user_selections_data)

    if st.button('Export to CSV'):
        export_df = room_names_df.merge(user_selections_df, on='Original Room Name', how='left')
        export_df.rename(columns={'User Selection': 'Selected Boring Name', 'Top Suggestion': 'Suggested Boring Name'}, inplace=True)
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download updated names CSV", data=csv, file_name="updated_names.csv", mime="text/csv")
