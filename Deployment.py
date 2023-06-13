import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# Load the trained models
DTClassifier = pickle.load(open('DecisionTree.pkl', 'rb'))
RFClassifier = pickle.load(open('RandomForest.pkl', 'rb'))
SVM = pickle.load(open('SVM.pkl', 'rb'))
KNN = pickle.load(open('KNN.pkl', 'rb'))
LRClassifier = pickle.load(open('LogisticRegression.pkl', 'rb'))
MLPC = pickle.load(open('MLPC.pkl', 'rb'))

# Streamlit App
@st.cache(suppress_st_warning=True)
def load_data():
    data = pd.read_csv("Train.csv", delimiter=";")
    return data

def main():
    # Set the title and description of the app
    st.title("Workplace Stress Prediction")
    data = load_data()
    st.subheader("Our Data")
    st.dataframe(data)
    st.write("This app predicts workplace stress based on user input.")
    if st.button("Predict Yours", key="Entry"):
        st.write("Enter Your Details")
        # Create input fields for user to enter features
        Age = st.text_input("Age", "28")
        Education = st.text_input("Education Level", "3")
        AvgDailyHours = st.text_input("Average Daily Hours", "8")
        EducationField = st.selectbox("Department", ["Technical Degree", "Medical", "Human Resources", "Marketing", "Other"])
        Department = st.selectbox("Education Field", ["Manufacturing Director", "Sales Representative", "Healthcare Representative", "Research Director", "Manager", "Human Resources", "Sales Executive", "Laboratory Technician", "Research Scientist"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        HasFlexibleTimings = st.selectbox("Has Flexible Timings", ["No", "Yes"])
        IsIndividualContributor = st.selectbox("Is Individual Contributor", ["Yes", "No"])
        JobInvolvement = st.slider("Job Involvement", 1, 4, 2)
        JobRole = st.selectbox("Job Role", ["Manager", "Engineer", "Analyst", "Sales", "HR"])
        JobSatisfaction = st.slider("Job Satisfaction", 1, 5, 3)
        LeavesTaken = st.text_input("Leaves Taken", "15")
        MaritalStatus = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
        MicromanagedAtWork = st.slider("Micromanaged at Work", 0, 10, 5)
        MonthlyIncome = st.text_input("Monthly Income", "25000")
        NumCompaniesWorked = st.text_input("Number of Companies Worked", "2")
        PercentSalaryHike = st.text_input("Percent Salary Hike", "45")
        PerformanceRating = st.slider("Performance Rating", 1, 5, 3)
        RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 5, 3)
        RemoteWorkSatisfaction = st.selectbox("Remote Work Satisfaction", ["Medium", "Very High", "High", "Low", "Very Low"])
        SelfMotivationLevel = st.slider("Self Motivation Level", 1, 5, 3)
        TotalWorkingYears = st.text_input("Total Working Years", "5")
        TrainingTimesLastYear = st.text_input("Training Times Last Year", "6")
        WorkLifeBalance = st.slider("Work-Life Balance", 1, 5, 3)
        WorkLoadLevel = st.selectbox("Work Load Level", ["Low", "High", "Medium"])
        YearsAtCompany = st.text_input("Years at Company", "2")
        YearsSinceLastPromotion = st.text_input("Years Since Last Promotion", "2")
        YearsWithCurrManager = st.text_input("Years with Current Manager", "2")
        UserData = pd.DataFrame({
            'EmployeeID': [000000],
            'Age': [Age],
            'AvgDailyHours': [AvgDailyHours],
            'Department': [Department],
            'Education': [Education],
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
            'RemoteWorkSatistfaction': [RemoteWorkSatisfaction],
            'SelfMotivationLevel': [SelfMotivationLevel],
            'TotalWorkingYears': [TotalWorkingYears],
            'TrainingTimesLastYear': [TrainingTimesLastYear],
            'WorkLifeBalance': [WorkLifeBalance],
            'WorkLoadLevel': [WorkLoadLevel],
            'YearsAtCompany': [YearsAtCompany],
            'YearsSinceLastPromotion': [YearsSinceLastPromotion],
            'YearsWithCurrManager': [YearsWithCurrManager]
        })
        st.write("Your Details")
        st.dataframe(UserData)
        le = LabelEncoder()
        for column in UserData.columns:
            if UserData[column].dtype == object:
                TempVal = le.fit_transform(UserData[column].astype('category'))
                UserData.drop(labels=[column], axis="columns", inplace=True)
                UserData[column] = TempVal
        st.dataframe(UserData)
        LRPredicted = LRClassifier.predict(UserData)
        # Display the predictions
        if st.button("Continue", key="continue"):
            st.subheader("Predictions")
            if LRPredicted == 0:
                st.write("You are a bit STRESSED! Take some assistance..")
            else:
                st.write("You are perfectly alright. Keep Rocking!!")


if __name__ == "__main__":
    main()
