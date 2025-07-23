import requests
import zipfile
import io
import os
import tempfile
import shutil
import re
from collections import defaultdict

class GitHubMLDetector:
    def __init__(self):
        self.ml_patterns = {
            'TensorFlow': ['tensorflow', 'tf.', '@tf.function', 'tf.keras', 'tf.Variable'],
            'PyTorch': ['torch.', 'torch.nn', '.backward()', 'torch.optim', 'torch.tensor'],
            'Scikit-learn': ['sklearn.', 'from sklearn', '.fit(', '.predict(', 'RandomForestClassifier'],
            'Pandas': ['pandas', 'pd.', 'DataFrame', 'read_csv', 'pd.read'],
            'NumPy': ['numpy', 'np.', 'np.array', 'np.zeros', 'np.ones'],
            'Keras': ['from keras', 'keras.layers', 'keras.models', 'Sequential()'],
            'XGBoost': ['xgboost', 'xgb.', 'XGBClassifier', 'XGBRegressor'],
            'Transformers': ['transformers', 'AutoModel', 'AutoTokenizer', 'pipeline'],
            'LightGBM': ['lightgbm', 'lgb.', 'LGBMClassifier'],
            'OpenCV': ['cv2', 'opencv', 'import cv2'],
            'Matplotlib': ['matplotlib', 'plt.', 'pyplot'],
            'Seaborn': ['seaborn', 'sns.', 'import seaborn']
        }
    
    def fetch_repo(self, repo_url, progress_callback=None):
        """Download GitHub repo with progress"""
        try:
            if progress_callback:
                progress_callback("Parsing GitHub URL...")
            
            # Clean and parse URL
            repo_url = repo_url.strip()
            if repo_url.endswith('.git'):
                repo_url = repo_url[:-4]
            
            parts = repo_url.replace('https://github.com/', '').split('/')
            if len(parts) < 2:
                return None, "Invalid GitHub URL format"
            
            owner, repo = parts[0], parts[1]
            
            # Try different branches
            branches = ['main', 'master', 'develop']
            
            for branch in branches:
                try:
                    download_url = f"https://github.com/{owner}/{repo}/archive/{branch}.zip"
                    
                    if progress_callback:
                        progress_callback(f"Downloading from {branch} branch...")
                    
                    response = requests.get(download_url, timeout=30)
                    if response.status_code == 200:
                        return self.extract_repo(response.content, progress_callback)
                except:
                    continue
            
            return None, "Repository not found or not accessible"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def extract_repo(self, zip_content, progress_callback=None):
        """Extract repo to temp directory"""
        temp_dir = tempfile.mkdtemp()
        try:
            if progress_callback:
                progress_callback("Extracting repository...")
            
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                zip_file.extractall(temp_dir)
            
            folders = [f for f in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, f))]
            if folders:
                return os.path.join(temp_dir, folders[0]), None
            return None, "No folders found"
        except Exception as e:
            return None, f"Extraction error: {str(e)}"
    
    def scan_files(self, repo_path, progress_callback=None):
        """Scan Python files and dependencies"""
        python_files = []
        dependencies = set()
        config_files = []
        
        if progress_callback:
            progress_callback("Scanning files...")
        
        total_files = 0
        for root, dirs, files in os.walk(repo_path):
            total_files += len(files)
        
        processed_files = 0
        
        for root, dirs, files in os.walk(repo_path):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.pytest_cache', 'venv', 'env']]
            
            for file in files:
                processed_files += 1
                if progress_callback and processed_files % 50 == 0:
                    progress = (processed_files / total_files) * 100
                    progress_callback(f"Scanning files... {progress:.1f}%")
                
                file_path = os.path.join(root, file)
                
                # Python files
                if file.endswith('.py'):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if content.strip() and len(content) > 50:  # Skip very small files
                                relative_path = os.path.relpath(file_path, repo_path)
                                python_files.append({
                                    'name': file,
                                    'path': relative_path,
                                    'content': content,
                                    'size': len(content)
                                })
                    except:
                        continue
                
                # Dependency files
                elif file in ['requirements.txt', 'setup.py', 'pyproject.toml', 'environment.yml', 'Pipfile']:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().lower()
                            config_files.append({'name': file, 'content': content})
                            
                            # Extract dependencies
                            for framework in self.ml_patterns.keys():
                                if framework.lower() in content:
                                    dependencies.add(framework)
                    except:
                        continue
        
        return python_files, list(dependencies), config_files
    
    def detect_frameworks(self, file_content):
        """Detect ML frameworks in code"""
        detected = {}
        content_lower = file_content.lower()
        
        for framework, patterns in self.ml_patterns.items():
            matches = 0
            matched_patterns = []
            
            for pattern in patterns:
                if pattern.lower() in content_lower:
                    matches += content_lower.count(pattern.lower())
                    matched_patterns.append(pattern)
            
            if matches > 0:
                detected[framework] = {
                    'count': matches,
                    'patterns': matched_patterns[:3]  # Limit to 3 patterns
                }
        
        return detected
    
    def calculate_confidence(self, framework_summary, total_files, ml_files_count):
        """Calculate detection confidence"""
        if not framework_summary:
            return 'None', 0
        
        max_count = max(framework_summary.values())
        file_ratio = ml_files_count / total_files if total_files > 0 else 0
        
        # Combined score
        score = (max_count * 0.7) + (file_ratio * 100 * 0.3)
        
        if score > 50:
            return 'High', score
        elif score > 20:
            return 'Medium', score
        elif score > 5:
            return 'Low', score
        else:
            return 'Very Low', score
    
    def analyze_repo(self, repo_url, progress_callback=None):
        """Main analysis function"""
        if progress_callback:
            progress_callback("Starting analysis...")
        
        # Download repo
        repo_path, error = self.fetch_repo(repo_url, progress_callback)
        if error:
            return {'error': error}
        
        try:
            # Scan files
            python_files, dependencies, config_files = self.scan_files(repo_path, progress_callback)
            
            if not python_files:
                return {'error': 'No Python files found in repository'}
            
            if progress_callback:
                progress_callback("Analyzing ML frameworks...")
            
            # Analyze each file
            framework_summary = defaultdict(int)
            ml_files = []
            file_details = []
            
            for i, file_info in enumerate(python_files):
                if progress_callback and i % 20 == 0:
                    progress = (i / len(python_files)) * 100
                    progress_callback(f"Analyzing files... {progress:.1f}%")
                
                detected = self.detect_frameworks(file_info['content'])
                if detected:
                    ml_files.append({
                        'file': file_info['name'],
                        'path': file_info['path'],
                        'frameworks': detected,
                        'size': file_info['size']
                    })
                    
                    for framework, details in detected.items():
                        framework_summary[framework] += details['count']
                
                # Store file details
                file_details.append({
                    'name': file_info['name'],
                    'path': file_info['path'],
                    'size': file_info['size'],
                    'has_ml': len(detected) > 0,
                    'frameworks': list(detected.keys())
                })
            
            # Calculate confidence
            confidence_level, confidence_score = self.calculate_confidence(
                framework_summary, len(python_files), len(ml_files)
            )
            
            # Get repository info
            repo_name = repo_url.split('/')[-1]
            repo_owner = repo_url.split('/')[-2]
            
            # Results
            ml_detected = len(framework_summary) > 0 or len(dependencies) > 0
            
            result = {
                'success': True,
                'repository_info': {
                    'name': repo_name,
                    'owner': repo_owner,
                    'url': repo_url
                },
                'ml_frameworks_detected': ml_detected,
                'frameworks_found': list(framework_summary.keys()),
                'dependencies_found': dependencies,
                'framework_summary': dict(framework_summary),
                'confidence': {
                    'level': confidence_level,
                    'score': round(confidence_score, 2)
                },
                'statistics': {
                    'total_files': len(python_files),
                    'ml_files_count': len(ml_files),
                    'total_size': sum(f['size'] for f in file_details),
                    'avg_file_size': sum(f['size'] for f in file_details) / len(file_details) if file_details else 0
                },
                'detailed_files': ml_files[:10],  # Top 10 ML files
                'all_files': file_details,
                'config_files': config_files
            }
            
            if progress_callback:
                progress_callback("Analysis complete!")
            
            return result
            
        finally:
            # Cleanup
            if repo_path and os.path.exists(repo_path):
                try:
                    shutil.rmtree(os.path.dirname(repo_path), ignore_errors=True)
                except:
                    pass

def detect_ml_frameworks(repo_url, progress_callback=None):
    """Simple function to detect ML frameworks"""
    detector = GitHubMLDetector()
    return detector.analyze_repo(repo_url, progress_callback)

def validate_github_url(url):
    """Validate GitHub URL format"""
    if not url:
        return False, "Please enter a GitHub URL"
    
    # Clean URL
    url = url.strip()
    
    # Check basic format
    if not url.startswith('https://github.com/'):
        return False, "URL must start with 'https://github.com/'"
    
    # Extract parts
    parts = url.replace('https://github.com/', '').replace('.git', '').split('/')
    
    if len(parts) < 2:
        return False, "URL must include owner and repository name"
    
    if not parts[0] or not parts[1]:
        return False, "Invalid owner or repository name"
    
    return True, "Valid GitHub URL"

# Example repositories for testing
EXAMPLE_REPOS = [
    "https://github.com/tensorflow/tensorflow",
    "https://github.com/pytorch/pytorch",
    "https://github.com/scikit-learn/scikit-learn",
    "https://github.com/pandas-dev/pandas",
    "https://github.com/huggingface/transformers",
    "https://github.com/microsoft/LightGBM",
    "https://github.com/dmlc/xgboost"
]