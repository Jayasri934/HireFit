import os
import re
import nltk
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import String 
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. App Initialization and Configuration ---
app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'your_super_secret_key_987' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' 
login_manager.login_message_category = 'info'

# Ensure necessary folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- 2. Database Model ---

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(String(20), unique=True, nullable=False)
    email = db.Column(String(120), unique=True, nullable=False)
    password_hash = db.Column(String(128))
    last_resume_text = db.Column(db.Text, nullable=True) 
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# --- 3. Data Loading and Skill Extraction Setup ---

def load_job_data(file_path='job_categories.csv'):
    """Loads job category data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        job_categories = {}
        all_skills = set()
        
        for index, row in df.iterrows():
            category = row['Category']
            skills_list = [s.strip().lower() for s in row['Skills'].split(',')]
            job_categories[category] = skills_list
            all_skills.update(skills_list)
            
        return job_categories, all_skills
        
    except FileNotFoundError:
        print(f"ERROR: Job categories file not found at {file_path}. Skill analysis will be limited.")
        return {}, set()
    except Exception as e:
        print(f"ERROR: Failed to load job data: {e}")
        return {}, set()

JOB_CATEGORIES, SKILL_KEYWORDS = load_job_data()

# NLTK Fix: Download NLTK resources robustly
STOPWORDS = set()
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    try:
        nltk.download('stopwords') 
    except Exception as e:
        print(f"Error during NLTK download: {e}. Running without stopwords.")

try:
    STOPWORDS = set(nltk.corpus.stopwords.words('english'))
    print("NLTK stopwords loaded successfully.")
except Exception:
    print("Warning: Could not load NLTK stopwords. Text preprocessing may be less effective.")


# --- 4. NLP/Resume Analysis Core Functions ---

def extract_text_from_pdf(pdf_path):
    """Extracts text content from a PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return None

def preprocess_text(text):
    """Cleans the text for analysis."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    tokens = [word for word in text.split() if word not in STOPWORDS and len(word) > 2]
    return ' '.join(tokens)

def calculate_match_score(resume_text, job_description_text):
    """Calculates the cosine similarity score (0-100) between two texts."""
    if not resume_text or not job_description_text:
        return 0.0

    cleaned_resume = preprocess_text(resume_text)
    cleaned_job_desc = preprocess_text(job_description_text)

    documents = [cleaned_resume, cleaned_job_desc]
    vectorizer = TfidfVectorizer()
    
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except ValueError:
        return 0.0

    return round(similarity_score * 100, 2) 


def extract_skills(resume_text):
    """Extracts relevant skills."""
    if not resume_text or not SKILL_KEYWORDS:
        return set()
    text = resume_text.lower()
    found_skills = set()
    for skill in SKILL_KEYWORDS:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            found_skills.add(skill.title()) 
    return found_skills

def suggest_jobs_by_category(extracted_skills):
    """Returns only the top 5 job suggestions."""
    suggestions = {}
    extracted_skills_lower = set(s.lower() for s in extracted_skills)

    for category, required_skills_list in JOB_CATEGORIES.items():
        required_skills_set = set(required_skills_list)
        matched_skills = extracted_skills_lower.intersection(required_skills_set)
        
        if required_skills_set:
            score = (len(matched_skills) / len(required_skills_set)) * 100
        else:
            score = 0
            
        suggestions[category] = round(score, 2)
        
    sorted_suggestions = sorted(suggestions.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_suggestions[:5] 


# --- 5. Routes for All Pages ---

@app.route('/')
@app.route('/home')
def home():
    """Home Page - The main landing page."""
    return render_template('home.html', title='Home')

@app.route('/about')
def about():
    """About Us Page."""
    return render_template('about.html', title='About Us')

@app.route('/jobs')
@login_required
def jobs():
    """Placeholder for a future Jobs page."""
    flash("This 'Jobs' feature is coming soon! For now, please use the Analysis page.", "info")
    return redirect(url_for('profile')) 

# --- Authentication Routes ---

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User Registration."""
    if current_user.is_authenticated:
        return redirect(url_for('profile'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = db.session.execute(db.select(User).filter_by(username=username)).scalar_one_or_none()
        email_check = db.session.execute(db.select(User).filter_by(email=email)).scalar_one_or_none()

        if user:
            flash('Username is already taken.', 'danger')
        elif email_check:
            flash('Email is already registered.', 'danger')
        else:
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash(f'Account created for {username}! You can now log in.', 'success')
            return redirect(url_for('login'))
        
    return render_template('register.html', title='Register')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User Login."""
    if current_user.is_authenticated:
        return redirect(url_for('profile'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = db.session.execute(db.select(User).filter_by(username=username)).scalar_one_or_none()

        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            flash('Logged in successfully!', 'success')
            return redirect(next_page or url_for('profile')) 
        else:
            flash('Login Unsuccessful. Check username and password', 'danger')
            
    return render_template('login.html', title='Login')

@app.route('/logout')
@login_required
def logout():
    """User Logout."""
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

@app.route('/clear_resume', methods=['POST'])
@login_required
def clear_resume():
    """Clears the user's last uploaded resume text from the database."""
    if current_user.last_resume_text:
        current_user.last_resume_text = None 
        db.session.commit()
        flash("Your previous resume data has been cleared. Please upload a new resume.", "info")
    else:
        flash("No previous resume found to clear.", "warning")
        
    return redirect(url_for('upload'))


@app.route('/profile')
@login_required
def profile():
    """User Profile/Dashboard (Main results page)."""
    
    resume_text = current_user.last_resume_text
    
    if not resume_text:
        flash("Please upload your resume first to activate your personalized skill analysis and job suggestions.", "warning")
        return redirect(url_for('upload'))
    
    user_skills = extract_skills(resume_text)
    job_suggestions = suggest_jobs_by_category(user_skills)
    
    return render_template('profile.html', 
                           title='Analysis', 
                           user_skills=user_skills, 
                           job_suggestions=job_suggestions,
                           raw_resume_text=resume_text)


@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Resume Upload and Specific Job Match Page. Fixes the Jinja2 UndefinedError."""
    
    # FIX: Initialize score to None for the initial GET request render
    score = None 
    
    if request.method == 'POST':
        file = request.files.get('resume_file')
        job_desc = request.form.get('job_description', '')

        if not file or file.filename == '' or not job_desc.strip():
            flash('Please select a PDF file and paste a Job Description.', 'danger')
            # If POST fails here, re-render the template with score=None
            return render_template('upload.html', title='Upload Resume', score=score)

        if file.filename.endswith('.pdf'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)
            
            resume_text = extract_text_from_pdf(filepath)
            
            if resume_text:
                current_user.last_resume_text = resume_text
                db.session.commit()

                score = calculate_match_score(resume_text, job_desc)
                
                flash(f'Analysis complete! Specific Job Match Score: {score}%. Skills and suggestions updated.', 'success')
                
                return redirect(url_for('profile')) 
            else:
                flash('Could not extract text from the PDF file. Try a different PDF.', 'danger')

            try:
                os.remove(filepath) 
            except OSError as e:
                print(f"Error removing file: {e}")
        
        else:
            flash('Invalid file type. Please upload a PDF file.', 'danger')
            
    # Render the upload form on GET request, passing the initialized score (None)
    # or if the POST request failed before redirecting.
    return render_template('upload.html', title='Upload Resume', score=score)


# --- 6. Database Creation (Run this once) ---
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    print("-" * 30)
    print(f"Loaded {len(JOB_CATEGORIES)} job categories from CSV.")
    print(f"Total {len(SKILL_KEYWORDS)} unique skills loaded.")
    print("-" * 30)
    app.run(debug=True)