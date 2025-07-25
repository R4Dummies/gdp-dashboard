import streamlit as st
import pandas as pd
import re
from typing import Dict, Set, List
import json
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Text Classification Tool",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state for dictionaries
if 'dictionaries' not in st.session_state:
    st.session_state.dictionaries = {
        'urgency_marketing': {
            'limited', 'limited time', 'limited run', 'limited edition', 'order now',
            'last chance', 'hurry', 'while supplies last', 'before they\'re gone',
            'selling out', 'selling fast', 'act now', 'don\'t wait', 'today only',
            'expires soon', 'final hours', 'almost gone'
        },
        'exclusive_marketing': {
            'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
            'members only', 'vip', 'special access', 'invitation only',
            'premium', 'privileged', 'limited access', 'select customers',
            'insider', 'private sale', 'early access'
        }
    }

def classify_text(text: str, dictionaries: Dict[str, Set[str]]) -> Dict[str, List[str]]:
    """
    Classify text using dictionary matching.
    Returns dict with category names as keys and matched terms as values.
    """
    if pd.isna(text) or not isinstance(text, str):
        return {}
    
    text_lower = text.lower()
    results = {}
    
    for category, terms in dictionaries.items():
        matches = []
        for term in terms:
            # Use word boundaries for exact matching
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            if re.search(pattern, text_lower):
                matches.append(term)
        
        if matches:
            results[category] = matches
    
    return results

def process_dataframe(df: pd.DataFrame, text_column: str, dictionaries: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    Process DataFrame and add classification results.
    """
    # Create a copy to avoid modifying the original
    df_results = df.copy()
    
    # Apply classification
    df_results['classifications'] = df_results[text_column].apply(
        lambda x: classify_text(x, dictionaries)
    )
    
    # Create binary columns for each category
    for category in dictionaries.keys():
        df_results[f'{category}_detected'] = df_results['classifications'].apply(
            lambda x: 1 if category in x else 0
        )
        df_results[f'{category}_matches'] = df_results['classifications'].apply(
            lambda x: ', '.join(x.get(category, []))
        )
    
    return df_results

def main():
    st.title("🔍 Text Classification Tool")
    st.markdown("Upload your dataset and customize dictionaries for text classification")
    
    # Sidebar for dictionary management
    st.sidebar.header("📚 Dictionary Management")
    
    # Dictionary editor
    st.sidebar.subheader("Edit Dictionaries")
    
    # Select dictionary to edit
    dict_names = list(st.session_state.dictionaries.keys())
    selected_dict = st.sidebar.selectbox("Select dictionary to edit:", dict_names)
    
    if selected_dict:
        st.sidebar.write(f"**{selected_dict}** terms:")
        
        # Display current terms
        current_terms = list(st.session_state.dictionaries[selected_dict])
        terms_text = '\n'.join(current_terms)
        
        # Text area for editing terms
        updated_terms = st.sidebar.text_area(
            "Edit terms (one per line):",
            value=terms_text,
            height=200,
            key=f"terms_{selected_dict}"
        )
        
        if st.sidebar.button(f"Update {selected_dict}"):
            # Update the dictionary
            new_terms = set()
            for term in updated_terms.split('\n'):
                term = term.strip()
                if term:
                    new_terms.add(term)
            st.session_state.dictionaries[selected_dict] = new_terms
            st.sidebar.success(f"Updated {selected_dict}!")
    
    # Add new dictionary
    st.sidebar.subheader("Add New Dictionary")
    new_dict_name = st.sidebar.text_input("Dictionary name:")
    new_dict_terms = st.sidebar.text_area("Terms (one per line):", height=100)
    
    if st.sidebar.button("Add Dictionary"):
        if new_dict_name and new_dict_terms:
            terms = set()
            for term in new_dict_terms.split('\n'):
                term = term.strip()
                if term:
                    terms.add(term)
            st.session_state.dictionaries[new_dict_name] = terms
            st.sidebar.success(f"Added {new_dict_name}!")
            st.rerun()
    
    # Delete dictionary
    if len(dict_names) > 1:
        dict_to_delete = st.sidebar.selectbox("Delete dictionary:", [""] + dict_names)
        if st.sidebar.button("Delete Dictionary") and dict_to_delete:
            del st.session_state.dictionaries[dict_to_delete]
            st.sidebar.success(f"Deleted {dict_to_delete}!")
            st.rerun()
    
    # Main content area
    st.header("📊 Data Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file containing text data to classify"
    )
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Show preview of the data
            st.subheader("📋 Data Preview")
            st.dataframe(df.head())
            
            # Select text column
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            if text_columns:
                selected_column = st.selectbox(
                    "Select the text column to classify:",
                    text_columns,
                    index=0 if 'Statement' not in text_columns else text_columns.index('Statement')
                )
                
                # Display current dictionaries
                st.subheader("📖 Current Dictionaries")
                
                col1, col2 = st.columns(2)
                dict_items = list(st.session_state.dictionaries.items())
                
                for i, (name, terms) in enumerate(dict_items):
                    with col1 if i % 2 == 0 else col2:
                        with st.expander(f"{name} ({len(terms)} terms)"):
                            st.write(", ".join(sorted(terms)))
                
                # Process button
                if st.button("🚀 Classify Text", type="primary"):
                    with st.spinner("Processing..."):
                        # Process the dataframe
                        df_results = process_dataframe(
                            df, selected_column, st.session_state.dictionaries
                        )
                        
                        # Display results
                        st.subheader("📈 Classification Results")
                        
                        # Summary statistics
                        st.write("**Summary:**")
                        summary_cols = st.columns(len(st.session_state.dictionaries))
                        
                        for i, category in enumerate(st.session_state.dictionaries.keys()):
                            with summary_cols[i]:
                                count = df_results[f'{category}_detected'].sum()
                                percentage = (count / len(df_results)) * 100
                                st.metric(
                                    label=category.replace('_', ' ').title(),
                                    value=f"{count}",
                                    delta=f"{percentage:.1f}%"
                                )
                        
                        # Show detailed results
                        st.subheader("📄 Detailed Results")
                        
                        # Filter options
                        filter_option = st.selectbox(
                            "Filter results:",
                            ["All rows", "Only classified rows", "Only unclassified rows"]
                        )
                        
                        if filter_option == "Only classified rows":
                            # Show only rows with at least one classification
                            mask = df_results[[f'{cat}_detected' for cat in st.session_state.dictionaries.keys()]].sum(axis=1) > 0
                            display_df = df_results[mask]
                        elif filter_option == "Only unclassified rows":
                            # Show only rows with no classifications
                            mask = df_results[[f'{cat}_detected' for cat in st.session_state.dictionaries.keys()]].sum(axis=1) == 0
                            display_df = df_results[mask]
                        else:
                            display_df = df_results
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Download button for results
                        csv_buffer = StringIO()
                        df_results.to_csv(csv_buffer, index=False)
                        csv_string = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv_string,
                            file_name="classified_results.csv",
                            mime="text/csv",
                            type="secondary"
                        )
                        
                        # Individual row inspection
                        st.subheader("🔍 Individual Row Inspection")
                        
                        if len(display_df) > 0:
                            row_index = st.selectbox(
                                "Select row to inspect:",
                                range(len(display_df)),
                                format_func=lambda x: f"Row {display_df.iloc[x].name}"
                            )
                            
                            if row_index is not None:
                                row = display_df.iloc[row_index]
                                
                                st.write(f"**Original Text:** {row[selected_column]}")
                                
                                # Show classifications for this row
                                detected = []
                                for category in st.session_state.dictionaries.keys():
                                    if row[f'{category}_detected']:
                                        matches = row[f'{category}_matches']
                                        detected.append(f"**{category}**: {matches}")
                                
                                if detected:
                                    st.write("**Detected Classifications:**")
                                    for detection in detected:
                                        st.write(f"- {detection}")
                                else:
                                    st.write("**No classifications detected**")
            else:
                st.error("No text columns found in the uploaded file.")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Show sample data format
        st.subheader("📝 Expected Data Format")
        sample_data = pd.DataFrame({
            'ID': [1, 2, 3],
            'Statement': [
                'Limited time offer! Get exclusive access now!',
                'Regular product description without marketing terms.',
                'Hurry! VIP members only - expires soon!'
            ]
        })
        st.dataframe(sample_data)

if __name__ == "__main__":
    main()
