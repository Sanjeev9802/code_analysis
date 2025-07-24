import streamlit as st
from utils import detect_ml_frameworks, validate_github_url

# Set page config
st.set_page_config(
    page_title="GitHub ML Framework Detector",
    page_icon="🤖",
    layout="wide"
)

def main():
    # Header
    st.title("🤖 GitHub ML Framework Detector")
    
    # URL input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        repo_url = st.text_input(
            "GitHub Repository URL:",
            placeholder="https://github.com/username/repository"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    # URL validation
    if repo_url:
        is_valid, validation_message = validate_github_url(repo_url)
        if not is_valid:
            st.error(f"❌ {validation_message}")
            return
    
    # Analysis
    if analyze_button and repo_url:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message):
            status_text.text(f"🔄 {message}")
        
        with st.spinner("Analyzing repository..."):
            try:
                result = detect_ml_frameworks(repo_url, progress_callback=update_progress)
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                return
        
        # Display results
        if 'error' in result:
            st.error(f"❌ {result['error']}")
            return
        
        st.markdown("---")
        
        # Main metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "ML Frameworks Detected",
                "Yes" if result['ml_frameworks_detected'] else "No"
            )
        
        with col2:
            st.metric(
                "Total Python Files",
                result['statistics']['total_files']
            )
        
        with col3:
            st.metric(
                "Repository Size",
                f"{result['statistics']['total_size'] / 1024:.1f} KB"
            )
        
        # Frameworks found
        if result['frameworks_found']:
            st.markdown("### 🎯 Frameworks Found")
            
            # Display frameworks as badges
            framework_html = ""
            for framework in result['frameworks_found']:
                count = result['framework_summary'].get(framework, 0)
                framework_html += f'''
                <span style="background-color: #e1f5fe; color: #01579b; padding: 0.25rem 0.5rem; 
                border-radius: 0.25rem; margin: 0.25rem; display: inline-block; font-weight: bold;">
                {framework} ({count})
                </span>
                '''
            
            st.markdown(framework_html, unsafe_allow_html=True)
        else:
            st.info("No ML frameworks detected in this repository.")

if __name__ == "__main__":
    main()