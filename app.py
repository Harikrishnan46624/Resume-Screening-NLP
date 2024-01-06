import streamlit as st
import pdfplumber
import pickle
import re
import nltk

# nltk.download('punkt')
# nltk.download('stopwords')

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            # Extract text from PDF file
            resume_text = extract_text_from_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred while extracting text: {e}")
            return

        # st.write("Original Resume Text:", resume_text)  # Print original text for debugging

        cleaned_resume = clean_resume(resume_text)
        # st.write("Cleaned Resume:", cleaned_resume)  # Print cleaned text for debugging

        input_features = tfidfd.transform([cleaned_resume]).toarray()
        # st.write("Input Features Shape:", input_features.shape)  # Print shape for debugging

        prediction_id = clf.predict(input_features)[0]
        st.write("Prediction ID:", prediction_id)  # Print prediction for debugging

        # Map category ID to category name
        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",  # Corrected mapping for category ID 6
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }


        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)

# Python main
if __name__ == "__main__":
    main()
