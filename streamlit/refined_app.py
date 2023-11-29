import pandas as pd
import streamlit as st
from transformers import pipeline
import os

st.set_page_config(layout="centered", page_title="The CSV Boring Machine", page_icon="🐗")

with st.container():
    st.title("🐗 The Boring CSV Machine")
    st.caption("Replace unique room names with the most similar boring name.")

def load_model():
    model_path = r'C:\dev\the-boring-labels\streamlit\training\finetunedmodel'
    # Using pipeline for zero-shot classification
    classifier = pipeline("zero-shot-classification", model=model_path)
    return classifier

classifier = load_model()

def calculate_predictions(room_name, candidate_labels):
    result = classifier(room_name, candidate_labels)
    top_class = result['labels'][0]
    top_score = result['scores'][0]
    return top_class, top_score

with st.sidebar:
    st.write("## Upload CSV Files")
    uploaded_classifier_file = st.file_uploader("Upload CSV with classifier/boring list", type=["csv"])
    uploaded_room_names_file = st.file_uploader("Upload CSV with original room names", type=["csv"])

if uploaded_classifier_file and uploaded_room_names_file:
    classifier_df = pd.read_csv(uploaded_classifier_file)
    room_names_df = pd.read_csv(uploaded_room_names_file)

    boring_names = classifier_df['Boring Name'].tolist()
    aggregated_df = room_names_df.groupby('Original Room Name').size().reset_index(name='Unique Count')

    annotations_data = []
    for _, row in aggregated_df.iterrows():
        original_name = row['Original Room Name']
        unique_count = row['Unique Count']
        top_class, top_score = calculate_predictions(original_name, boring_names)
        
        top_match_with_prob = f"{top_class} - {top_score*100:.0f}%"
        
        alphabetical_boring_names = sorted(boring_names)

        annotations_data.append({
            "Original Room Name": original_name,
            "Unique Count": unique_count,
            "Selected Boring Name": top_match_with_prob,
            "Top Boring Name Suggestion": top_class,
            "Boring Name Options": alphabetical_boring_names
        })

    df = pd.DataFrame(annotations_data)
    df_sorted = df.sort_values(by=['Unique Count', 'Original Room Name'], ascending=[False, True])
    df_for_editor = df_sorted[['Unique Count', 'Original Room Name', 'Selected Boring Name']]

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

        if edited_df is not None:
            df['Selected Boring Name'] = edited_df['Selected Boring Name']

        final_df = room_names_df.merge(
            df[['Original Room Name', 'Selected Boring Name', 'Top Boring Name Suggestion']],
            on='Original Room Name',
            how='left'
        )

        final_csv_df = final_df[['Original Room Name', 'Selected Boring Name', 'Top Boring Name Suggestion']]

        st.download_button(
            "⬇️ Download annotations as .csv",
            final_csv_df.to_csv(index=False),
            "annotated_room_names.csv",
            mime='text/csv',
            use_container_width=True
        )
