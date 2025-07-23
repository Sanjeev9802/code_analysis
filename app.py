import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
from utils import detect_ml_frameworks, validate_github_url, EXAMPLE_REPOS

# Set page config
st.set_page_config(
    page_title="GitHub ML Framework Detector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
#demo
# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.framework-badge {
    background-color: #e1f5fe;
    color: #01579b;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    margin: 0.25rem;
    display: inline-block;
    font-weight: bold;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 0.25rem;
    margin: 1rem 0;
}
.error-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 1rem;
    border-radius: 0.25rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 GitHub ML Framework Detector</h1>', unsafe_allow_html=True)
    st.markdown("**Analyze any GitHub repository to detect Machine Learning frameworks and libraries**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Example repositories
        st.subheader("📚 Try Example Repositories")
        selected_example = st.selectbox(
            "Select an example:",
            [""] + EXAMPLE_REPOS,
            format_func=lambda x: "Choose an example..." if x == "" else x.split('/')[-1]
        )
        
        if st.button("Use Selected Example"):
            if selected_example:
                st.session_state.repo_url = selected_example
        
        st.markdown("---")
        
        # Analysis options
        st.subheader("⚙️ Analysis Options")
        show_file_details = st.checkbox("Show detailed file analysis", value=True)
        show_visualizations = st.checkbox("Show visualizations", value=True)
        show_code_snippets = st.checkbox("Show code snippets", value=False)
        
        st.markdown("---")
        
        # Instructions
        st.subheader("📖 How to Use")
        st.markdown("""
        1. Enter a GitHub repository URL
        2. Click 'Analyze Repository'
        3. View the ML framework detection results
        4. Explore detailed file analysis
        """)
        
        st.markdown("---")
        
        # About
        st.subheader("ℹ️ About")
        st.markdown("""
        This tool analyzes GitHub repositories to detect:
        - Machine Learning frameworks
        - Data science libraries
        - Dependencies and requirements
        - Code patterns and usage
        """)
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # URL input
        repo_url = st.text_input(
            "🔗 GitHub Repository URL:",
            value=st.session_state.get('repo_url', ''),
            placeholder="https://github.com/username/repository",
            help="Enter the full GitHub repository URL"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        analyze_button = st.button(
            "🔍 Analyze Repository",
            type="primary",
            use_container_width=True
        )
    
    # URL validation
    if repo_url:
        is_valid, validation_message = validate_github_url(repo_url)
        if not is_valid:
            st.error(f"❌ {validation_message}")
            return
        else:
            st.success(f"✅ {validation_message}")
    
    # Analysis
    if analyze_button and repo_url:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message):
            status_text.text(f"🔄 {message}")
        
        # Perform analysis
        with st.spinner("Analyzing repository..."):
            try:
                result = detect_ml_frameworks(repo_url, progress_callback=update_progress)
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Store results in session state
                st.session_state.analysis_result = result
                st.session_state.analysis_timestamp = datetime.now()
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                return
    
    # Display results
    if 'analysis_result' in st.session_state:
        result = st.session_state.analysis_result
        
        if 'error' in result:
            st.error(f"❌ {result['error']}")
            return
        
        # Clear progress indicators
        if 'progress_bar' in locals():
            progress_bar.empty()
            status_text.empty()
        
        st.markdown("---")
        
        # Results header
        repo_info = result['repository_info']
        st.markdown(f"## 📊 Analysis Results for [{repo_info['name']}]({repo_info['url']})")
        st.markdown(f"**Owner:** {repo_info['owner']} | **Analyzed:** {st.session_state.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "ML Frameworks Detected",
                "Yes" if result['ml_frameworks_detected'] else "No",
                delta=f"{len(result['frameworks_found'])} frameworks" if result['frameworks_found'] else "None"
            )
        
        with col2:
            st.metric(
                "Total Python Files",
                result['statistics']['total_files'],
                delta=f"{result['statistics']['ml_files_count']} with ML"
            )
        
        with col3:
            st.metric(
                "Detection Confidence",
                result['confidence']['level'],
                delta=f"{result['confidence']['score']}% score"
            )
        
        with col4:
            st.metric(
                "Repository Size",
                f"{result['statistics']['total_size'] / 1024:.1f} KB",
                delta=f"Avg: {result['statistics']['avg_file_size']:.0f} chars/file"
            )
        
        # Frameworks found
        if result['frameworks_found']:
            st.markdown("### 🎯 Detected ML Frameworks")
            
            # Create framework badges
            framework_html = ""
            for framework in result['frameworks_found']:
                count = result['framework_summary'].get(framework, 0)
                framework_html += f'<span class="framework-badge">{framework} ({count})</span>'
            
            st.markdown(framework_html, unsafe_allow_html=True)
            
            # Dependencies
            if result['dependencies_found']:
                st.markdown("### 📦 Dependencies Found")
                deps_html = ""
                for dep in result['dependencies_found']:
                    deps_html += f'<span class="framework-badge">{dep}</span>'
                st.markdown(deps_html, unsafe_allow_html=True)
        
        # Visualizations
        if show_visualizations and result['frameworks_found']:
            st.markdown("### 📈 Framework Usage Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Framework distribution pie chart
                framework_data = result['framework_summary']
                if framework_data:
                    fig_pie = px.pie(
                        values=list(framework_data.values()),
                        names=list(framework_data.keys()),
                        title="Framework Distribution"
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Framework usage bar chart
                if framework_data:
                    fig_bar = px.bar(
                        x=list(framework_data.values()),
                        y=list(framework_data.keys()),
                        orientation='h',
                        title="Framework Usage Count",
                        labels={'x': 'Usage Count', 'y': 'Framework'}
                    )
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # File analysis chart
            if result['all_files']:
                file_data = pd.DataFrame(result['all_files'])
                
                # ML vs Non-ML files
                ml_count = len(file_data[file_data['has_ml'] == True])
                non_ml_count = len(file_data[file_data['has_ml'] == False])
                
                fig_files = go.Figure(data=[
                    go.Bar(name='ML Files', x=['Files'], y=[ml_count], marker_color='lightblue'),
                    go.Bar(name='Non-ML Files', x=['Files'], y=[non_ml_count], marker_color='lightgray')
                ])
                fig_files.update_layout(
                    title='ML vs Non-ML Files',
                    barmode='stack',
                    yaxis_title='Number of Files'
                )
                st.plotly_chart(fig_files, use_container_width=True)
        
        # Detailed file analysis
        if show_file_details and result['detailed_files']:
            st.markdown("### 📄 Detailed File Analysis")
            
            for i, file_info in enumerate(result['detailed_files'][:5]):  # Show top 5
                with st.expander(f"📝 {file_info['file']} ({file_info['size']} chars)"):
                    st.markdown(f"**Path:** `{file_info['path']}`")
                    
                    # Frameworks in this file
                    for framework, details in file_info['frameworks'].items():
                        st.markdown(f"**{framework}:**")
                        st.markdown(f"- Usage count: {details['count']}")
                        st.markdown(f"- Patterns found: {', '.join(details['patterns'])}")
                    
                    # Show code snippet if requested
                    if show_code_snippets and i == 0:  # Only for first file
                        st.markdown("**Code Preview:**")
                        # This would require storing file content in results
                        st.code("# Code preview feature - implement based on needs", language="python")
        
        # File overview table
        if result['all_files']:
            st.markdown("### 📋 File Overview")
            
            file_df = pd.DataFrame(result['all_files'])
            file_df['ML Frameworks'] = file_df['frameworks'].apply(lambda x: ', '.join(x) if x else 'None')
            file_df['Size (KB)'] = (file_df['size'] / 1024).round(2)
            file_df['Has ML'] = file_df['has_ml'].apply(lambda x: '✅' if x else '❌')
            
            display_df = file_df[['name', 'Has ML', 'ML Frameworks', 'Size (KB)']].copy()
            display_df.columns = ['File Name', 'Contains ML', 'Frameworks', 'Size (KB)']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        
        # Download results
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON download
            import json
            result_json = json.dumps(result, indent=2, default=str)
            st.download_button(
                label="📥 Download JSON Report",
                data=result_json,
                file_name=f"{repo_info['name']}_ml_analysis.json",
                mime="application/json"
            )
        
        with col2:
            # CSV download
            if result['all_files']:
                csv_data = pd.DataFrame(result['all_files']).to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv_data,
                    file_name=f"{repo_info['name']}_file_analysis.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 GitHub ML Framework Detector | Built with Streamlit</p>
        <p>Analyze any public GitHub repository to detect ML frameworks and patterns</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()