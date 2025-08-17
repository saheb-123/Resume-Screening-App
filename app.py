
import nltk
import re
import streamlit as st
import pickle
import pandas as pd
nltk.download('punkt')
nltk.download('stopwords')

#loading the model and vectorizer
model = pickle.load(open('knn_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))   


def clean_resume(resume_text):
    cleaned_text = re.sub(r'\b[\w.%+-]+@gmail\.com\b', '', resume_text)
    cleaned_text = re.sub(r'\d+', '', cleaned_text)  # Remove numbers
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra spaces
    cleaned_text = re.sub(r'@\s+\s', '', cleaned_text)  # Remove email addresses
    cleaned_text = re.sub(r'http\S+|www\S+|https\S+', '', cleaned_text)  # Remove URLs
    cleaned_text = re.sub(r'\W+', ' ', cleaned_text)  # Remove special characters
    cleaned_text = re.sub(r'RT|cc',' ',cleaned_text)
    cleaned_text = cleaned_text.lower()  # Convert to lowercase
    return cleaned_text.strip()
#web app 
def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except Exception as e:
            #if utf-8 fails, try latin-1 encoding
            resume_text = resume_bytes.decode('latin-1')
            
        cleaned_resume = clean_resume(resume_text)
        
        # Predict the category
        input_features = vectorizer.transform([cleaned_resume])
        prediction_id = model.predict(input_features)[0]
        st.write("Predicted Category:", le.inverse_transform([prediction_id])[0])
        
        category_mapping = {
        15: "Java Developer",
        23: "Testing",
        8: "DevOps Engineer",
        20: "Python Developer",
        24: "Web Developer",
        12: "HR",
        13: "Hadoop",
        3: "Blockchain",
        10: "ETL Developer",
        18: "Operation Manager",
        6: "Data Scientist",
        22: "Sales",
        16: "Mechanical Engineer",
        1: "Arts Graduate",
        7: "Database",
        11: "Electrical Engineer",
        14: "Health and Finance",
        19: "PMO",
        4: "Business Analyst",
        9: "DotNet Developer",
        2: "Automation Testing",
        17: "Network Engineer",
        21: "SAP Developer",
        5: "Civil Engineer",
        0: "Advocate",
    }
       
        
if __name__ == "__main__":
    main()
    