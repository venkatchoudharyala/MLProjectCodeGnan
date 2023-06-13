import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained models
DTClassifier = pickle.load(open('DecisionTree', 'rb'))
RFClassifier = pickle.load(open('RandomForest.pkl', 'rb'))
SVM = pickle.load(open('SVM.pkl', 'rb'))
KNN = pickle.load(open('KNN.pkl', 'rb'))
LRClassifier = pickle.load(open('LogisticRegression.pkl', 'rb'))
MLPC= pickle.load(open('MLPC.pkl', 'rb'))

# Streamlit App
def main():
  # Set the title and description of the app
  st.title("Workplace Stress Prediction")
  st.write("This app predicts workplace stress based on user input.")
  if st.button("Predict Your's"):
    # Create input fields for user to enter features
    Age = st.text_input("Age", "28")
    Education = st.selectbox("Education Level", ["High School", "Bachelor's Degree", "Master's Degree", "Ph.D."])
    AvgDailyHours = st.text_input("Average Daily Hours", "8")
    Department = st.selectbox("Department", ["Sales", "Marketing", "Finance", "HR", "Operations"])
    EducationField = st.selectbox("Education Field", ["Mathematics", "Science", "Arts", "Engineering"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    HasFlexibleTimings = st.selectbox("Has Flexible Timings", ["Yes", "No"])
    IsIndividualContributor = st.selectbox("Is Individual Contributor", ["Yes", "No"])
    JobInvolvement = st.slider("Job Involvement", 1, 4, 2)
    JobRole = st.selectbox("Job Role", ["Manager", "Engineer", "Analyst", "Sales", "HR"])
    JobSatisfaction = st.slider("Job Satisfaction", 1, 5, 3)
    LeavesTaken = st.text_input("Leaves Taken", "15")
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    MicromanagedAtWork = st.slider("Micromanaged at Work", 0, 10, 5)
    MonthlyIncome = st.text_input("Monthly Income", "25000")
    NumCompaniesWorked = st.text_input("Number of Companies Worked", "2")
    PercentSalaryHike = st.text_input("Percent Salary Hike", "45")
    PerformanceRating = st.slider("Performance Rating", 1, 5, 3)
    RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 5, 3)
    RemoteWorkSatisfaction = st.selectbox("Remote Work Satisfaction", ["Satisfied", "Neutral", "Unsatisfied"])
    SelfMotivationLevel = st.slider("Self Motivation Level", 1, 5, 3)
    TotalWorkingYears = st.text_input("Total Working Years", "5")
    TrainingTimesLastYear = st.text_input("Training Times Last Year", "6")
    WorkLifeBalance = st.slider("Work-Life Balance", 1, 5, 3)
    WorkLoadLevel = st.selectbox("Work Load Level", ["Low", "Medium", "High"])
    YearsAtCompany = st.text_input("Years at Company", "2")
    YearsSinceLastPromotion = st.text_input("Years Since Last Promotion", "2")
    YearsWithCurrManager = st.text_input("Years with Current Manager", "2")

    UserData = pd.DataFrame({
    'Age': [Age],
    'Education': [Education],
    'AvgDailyHours': [AvgDailyHours],
    'Department': [Department],
    'EducationField': [EducationField],
    'Gender': [Gender],
    'HasFlexibleTimings': [HasFlexibleTimings],
    'IsIndividualContributor': [IsIndividualContributor],
    'JobInvolvement': [JobInvolvement],
    'JobRole': [JobRole],
    'JobSatisfaction': [JobSatisfaction],
    'LeavesTaken': [LeavesTaken],
    'MaritalStatus': [MaritalStatus],
    'MicromanagedAtWork': [MicromanagedAtWork],
    'MonthlyIncome': [MonthlyIncome],
    'NumCompaniesWorked': [NumCompaniesWorked],
    'PercentSalaryHike': [PercentSalaryHike],
    'PerformanceRating': [PerformanceRating],
    'RelationshipSatisfaction': [RelationshipSatisfaction],
    'RemoteWorkSatisfaction': [RemoteWorkSatisfaction],
    'SelfMotivationLevel': [SelfMotivationLevel],
    'TotalWorkingYears': [TotalWorkingYears],
    'TrainingTimesLastYear': [TrainingTimesLastYear],
    'WorkLifeBalance': [WorkLifeBalance],
    'WorkLoadLevel': [WorkLoadLevel],
    'YearsAtCompany': [YearsAtCompany],
    'YearsSinceLastPromotion': [YearsSinceLastPromotion],
    'YearsWithCurrManager': [YearsWithCurrManager]})

    le = LabelEncoder()
    for column in UserData.columns:
      TempVal = le.fit_transform(UserData[column].astype('category'))
      UserData.drop(labels=[column], axis="columns", inplace=True)
      UserData[column] = TempVal
    
    LRPredicted = LRClassifier.predict(UserData)

    # Display the predictions
    st.subheader("Predictions")
    if(LRPredicted == 0):
      st.write("You are a bit STRESSED!, take some assistance..")
    else:
      st.write("You are perfectly alright keep Rocking!!")

if __name__ == "__main__":
    main()
