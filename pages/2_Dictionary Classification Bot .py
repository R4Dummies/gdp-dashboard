import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import List, Dict, Tuple, Optional
import re

def main():
    st.set_page_config(
        page_title="Dictionary Classification Bot",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Dictionary Classification Bot")
    st.markdown("**Enter keywords and classify statements to analyze their effectiveness**")
    
    # Initialize session state
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None
    if 'dictionary' not in st.session_state:
        st.session_state.dictionary = []
    if 'classification_results' not in st.session_state:
        st.session_state.classification_results = None
    if 'overall_metrics' not in st.session_state:
        st.session_state.overall_metrics = None
    if 'keyword_analysis' not in st.session_state:
        st.session_state.keyword_analysis = None

    # Step 1: Input CSV Data
    st.header("📊 Step 1: Input Sample Data")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.csv_data = df
                st.success(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
    
    with col2:
        st.subheader("Or Use Sample Data")
        if st.button("Load Sample Data"):
            sample_data = {
                'ID': [1, 2, 3, 4, 5],
                'Statement': [
                    "It's SPRING TRUNK SHOW week!",
                    "I am offering 4 shirts styled the way you want & the 5th is Also tossing in MAGNETIC COLLAR STAY to help keep your collars in place!",
                    "In recognition of Earth Day, I would like to showcase our collection of Earth Fibers!",
                    "It is now time to do some \"wardrobe crunches,\" and check your basics! Never on sale.",
                    "He's a hard worker and always willing to lend a hand. The prices are the best I've seen in 17 years of servicing my clients."
                ],
                'Answer': [1, 1, 0, 1, 0]
            }
            st.session_state.csv_data = pd.DataFrame(sample_data)
            st.success("✅ Sample data loaded")
            st.dataframe(st.session_state.csv_data)
    
    # Column Selection
    if st.session_state.csv_data is not None:
        st.subheader("📋 Column Selection")
        col1, col2 = st.columns(2)
        
        with col1:
            text_column = st.selectbox(
                "Select Text Column for Analysis",
                options=st.session_state.csv_data.columns.tolist(),
                key="text_column"
            )
        
        with col2:
            ground_truth_options = ["None (Optional)"] + st.session_state.csv_data.columns.tolist()
            ground_truth_column = st.selectbox(
                "Select Ground Truth Column (0/1 values)",
                options=ground_truth_options,
                key="ground_truth_column"
            )
            if ground_truth_column == "None (Optional)":
                ground_truth_column = None

    # Step 2: Dictionary Management
    st.header("📖 Step 2: Keyword Dictionary Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Default Dictionary")
        default_keywords = [
            "custom", "customized", "customization", "personalized", "bespoke",
            "tailored", "made-to-order", "handcrafted", "unique", "exclusive"
        ]
        
        # Allow users to modify the default dictionary
        dictionary_text = st.text_area(
            "Edit Keywords (one per line or comma-separated)",
            value="\n".join(st.session_state.dictionary if st.session_state.dictionary else default_keywords),
            height=200,
            help="Enter keywords one per line or separate with commas"
        )
        
        if st.button("Update Dictionary"):
            # Parse keywords - handle both line-separated and comma-separated
            if '\n' in dictionary_text:
                keywords = [k.strip().strip('"\'') for k in dictionary_text.split('\n') if k.strip()]
            else:
                keywords = [k.strip().strip('"\'') for k in dictionary_text.split(',') if k.strip()]
            
            st.session_state.dictionary = [k for k in keywords if k]
            st.success(f"✅ Dictionary updated with {len(st.session_state.dictionary)} keywords")
    
    with col2:
        st.subheader("Current Dictionary")
        if st.session_state.dictionary:
            for i, keyword in enumerate(st.session_state.dictionary, 1):
                st.write(f"{i}. {keyword}")
        else:
            st.write("No keywords in dictionary")

    # Step 3: Classification
    if (st.session_state.csv_data is not None and 
        st.session_state.dictionary and 
        'text_column' in st.session_state):
        
        st.header("🎯 Step 3: Classify Statements")
        
        if st.button("🚀 Classify Statements", type="primary"):
            with st.spinner("Classifying statements..."):
                results = classify_statements(
                    st.session_state.csv_data,
                    st.session_state.dictionary,
                    st.session_state.text_column,
                    st.session_state.get('ground_truth_column')
                )
                st.session_state.classification_results = results['results']
                st.session_state.overall_metrics = results['metrics']
            
            st.success("✅ Classification completed!")

    # Step 4: Results Display
    if st.session_state.classification_results is not None:
        display_classification_results()

    # Step 5: Keyword Analysis
    if (st.session_state.classification_results is not None and 
        st.session_state.get('ground_truth_column')):
        
        st.header("📈 Step 4: Keyword Impact Analysis")
        
        if st.button("🔍 Analyze Keyword Impact"):
            with st.spinner("Analyzing keywords..."):
                analysis = analyze_keywords(
                    st.session_state.csv_data,
                    st.session_state.dictionary,
                    st.session_state.text_column,
                    st.session_state.ground_truth_column
                )
                st.session_state.keyword_analysis = analysis
            
            st.success("✅ Keyword analysis completed!")
        
        if st.session_state.keyword_analysis:
            display_keyword_analysis()

def classify_statements(df: pd.DataFrame, dictionary: List[str], text_column: str, ground_truth_column: Optional[str]) -> Dict:
    """Classify statements based on dictionary keywords"""
    results = []
    true_positives = false_positives = false_negatives = true_negatives = 0
    
    for _, row in df.iterrows():
        statement = str(row[text_column]).lower()
        
        # Find matched keywords
        matched_keywords = [keyword for keyword in dictionary 
                          if keyword.lower() in statement]
        
        predicted = 1 if matched_keywords else 0
        ground_truth = None
        category = None
        
        if ground_truth_column and ground_truth_column in df.columns:
            try:
                ground_truth_value = row[ground_truth_column]
                if not pd.isna(ground_truth_value):
                    ground_truth = int(float(ground_truth_value))  # Handle both int and float strings
                    
                    # Calculate confusion matrix
                    if predicted == 1 and ground_truth == 1:
                        true_positives += 1
                        category = 'TP'
                    elif predicted == 1 and ground_truth == 0:
                        false_positives += 1
                        category = 'FP'
                    elif predicted == 0 and ground_truth == 1:
                        false_negatives += 1
                        category = 'FN'
                    else:
                        true_negatives += 1
                        category = 'TN'
            except (ValueError, TypeError):
                # Keep ground_truth as None if conversion fails
                pass
        
        result = {
            **row.to_dict(),
            'predicted': predicted,
            'ground_truth': ground_truth,
            'category': category,
            'matched_keywords': matched_keywords,
            'keyword_count': len(matched_keywords)
        }
        results.append(result)
    
    # Calculate metrics
    metrics = None
    if ground_truth_column:
        total = true_positives + false_positives + false_negatives + true_negatives
        if total > 0:
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (true_positives + true_negatives) / total
            
            metrics = {
                'precision': precision * 100,
                'recall': recall * 100,
                'f1_score': f1 * 100,
                'accuracy': accuracy * 100,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'true_negatives': true_negatives
            }
    
    return {'results': results, 'metrics': metrics}

def display_classification_results():
    """Display classification results and metrics"""
    st.header("📊 Classification Results")
    
    # Overall metrics
    if st.session_state.overall_metrics:
        st.subheader("📈 Overall Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{st.session_state.overall_metrics['accuracy']:.1f}%")
        with col2:
            st.metric("Precision", f"{st.session_state.overall_metrics['precision']:.1f}%")
        with col3:
            st.metric("Recall", f"{st.session_state.overall_metrics['recall']:.1f}%")
        with col4:
            st.metric("F1 Score", f"{st.session_state.overall_metrics['f1_score']:.1f}%")
        
        # Confusion matrix
        st.subheader("🔢 Confusion Matrix")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("True Positives", st.session_state.overall_metrics['true_positives'])
        with col2:
            st.metric("False Positives", st.session_state.overall_metrics['false_positives'])
        with col3:
            st.metric("False Negatives", st.session_state.overall_metrics['false_negatives'])
        with col4:
            st.metric("True Negatives", st.session_state.overall_metrics['true_negatives'])
    
    # Results table
    st.subheader("📋 Detailed Results")
    results_df = pd.DataFrame(st.session_state.classification_results)
    
    # Add filters
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_filter = st.selectbox(
            "Filter by Prediction",
            options=["All", "Positive (1)", "Negative (0)"]
        )
    
    with col2:
        if 'category' in results_df.columns:
            category_filter = st.selectbox(
                "Filter by Category",
                options=["All", "TP", "FP", "FN", "TN"]
            )
        else:
            category_filter = "All"
    
    # Apply filters
    filtered_df = results_df.copy()
    
    if prediction_filter == "Positive (1)":
        filtered_df = filtered_df[filtered_df['predicted'] == 1]
    elif prediction_filter == "Negative (0)":
        filtered_df = filtered_df[filtered_df['predicted'] == 0]
    
    if category_filter != "All" and 'category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['category'] == category_filter]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Export options
    st.subheader("💾 Export Results")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Classification Results CSV"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.session_state.overall_metrics and st.button("📥 Download Metrics CSV"):
            metrics_data = []
            for key, value in st.session_state.overall_metrics.items():
                metrics_data.append({"Metric": key, "Value": value})
            
            metrics_df = pd.DataFrame(metrics_data)
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name="classification_metrics.csv",
                mime="text/csv"
            )

def analyze_keywords(df: pd.DataFrame, dictionary: List[str], text_column: str, ground_truth_column: str) -> Dict:
    """Analyze individual keyword performance"""
    keyword_metrics = []
    
    for keyword in dictionary:
        true_positives = []
        false_positives = []
        total_positives = 0
        
        for _, row in df.iterrows():
            try:
                statement = str(row[text_column]).lower()
                contains_keyword = keyword.lower() in statement
                
                # Safely convert ground truth to integer
                ground_truth_value = row[ground_truth_column]
                if pd.isna(ground_truth_value):
                    continue  # Skip rows with missing ground truth
                
                ground_truth = int(float(ground_truth_value))  # Handle both int and float strings
                
                if ground_truth == 1:
                    total_positives += 1
                    if contains_keyword:
                        true_positives.append(row.to_dict())
                elif ground_truth == 0 and contains_keyword:
                    false_positives.append(row.to_dict())
                    
            except (ValueError, TypeError, KeyError):
                # Skip rows where conversion fails
                continue
        
        # Calculate metrics
        recall = len(true_positives) / total_positives if total_positives > 0 else 0
        precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
        
        keyword_metrics.append({
            'keyword': keyword,
            'true_positives_count': len(true_positives),
            'false_positives_count': len(false_positives),
            'recall': recall * 100,
            'precision': precision * 100,
            'f1_score': f1 * 100,
            'true_positive_examples': true_positives[:3],
            'false_positive_examples': false_positives[:3]
        })
    
    # Sort by different metrics
    by_recall = sorted(keyword_metrics, key=lambda x: x['recall'], reverse=True)[:10]
    by_precision = sorted(keyword_metrics, key=lambda x: x['precision'], reverse=True)[:10]
    by_f1 = sorted(keyword_metrics, key=lambda x: x['f1_score'], reverse=True)[:10]
    
    return {
        'by_recall': by_recall,
        'by_precision': by_precision,
        'by_f1': by_f1
    }

def display_keyword_analysis():
    """Display keyword analysis results"""
    st.subheader("🔍 Keyword Performance Analysis")
    
    # Metric selection
    metric_type = st.selectbox(
        "Select Metric for Analysis",
        options=["Recall", "Precision", "F1 Score"],
        help={
            "Recall": "Percentage of true positive cases captured",
            "Precision": "Percentage of predictions that are correct", 
            "F1 Score": "Harmonic mean of precision and recall"
        }
    )
    
    # Get data based on selected metric
    if metric_type == "Recall":
        data = st.session_state.keyword_analysis['by_recall']
        color = "green"
    elif metric_type == "Precision":
        data = st.session_state.keyword_analysis['by_precision']
        color = "blue"
    else:
        data = st.session_state.keyword_analysis['by_f1']
        color = "purple"
    
    # Display keyword analysis
    for i, analysis in enumerate(data):
        with st.expander(f"#{i+1} {analysis['keyword']} - {metric_type}: {analysis[metric_type.lower().replace(' ', '_')]:.1f}%"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Recall", f"{analysis['recall']:.1f}%")
            with col2:
                st.metric("Precision", f"{analysis['precision']:.1f}%")
            with col3:
                st.metric("F1 Score", f"{analysis['f1_score']:.1f}%")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"✅ True Positives ({analysis['true_positives_count']})")
                if analysis['true_positive_examples']:
                    for example in analysis['true_positive_examples']:
                        st.write(f"• {example[st.session_state.text_column]}")
                else:
                    st.write("No examples")
            
            with col2:
                st.subheader(f"❌ False Positives ({analysis['false_positives_count']})")
                if analysis['false_positive_examples']:
                    for example in analysis['false_positive_examples']:
                        st.write(f"• {example[st.session_state.text_column]}")
                else:
                    st.write("No examples")
    
    # Export keyword analysis
    if st.button("📥 Download Keyword Analysis CSV"):
        analysis_data = []
        for i, analysis in enumerate(data):
            analysis_data.append({
                'Rank': i + 1,
                'Keyword': analysis['keyword'],
                'Recall_%': analysis['recall'],
                'Precision_%': analysis['precision'],
                'F1_Score_%': analysis['f1_score'],
                'True_Positives_Count': analysis['true_positives_count'],
                'False_Positives_Count': analysis['false_positives_count']
            })
        
        analysis_df = pd.DataFrame(analysis_data)
        csv = analysis_df.to_csv(index=False)
        st.download_button(
            label="Click to Download",
            data=csv,
            file_name=f"keyword_analysis_{metric_type.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
