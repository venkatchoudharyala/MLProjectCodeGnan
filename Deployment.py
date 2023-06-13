import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained models
DTClassifier = pickle.load(open('DecisionTree.pkl', 'rb'))
RFClassifier = pickle.load(open('RandomForest.pkl', 'rb'))
SVM = pickle.load(open('SVM.pkl', 'rb'))
KNN = pickle.load(open('KNN.pkl', 'rb'))
LRClassifier = pickle.load(open('LogisticRegression.pkl', 'rb'))
MLPC = pickle.load(open('MLPC.pkl', 'rb'))

# Load the data
@st.cache(suppress_st_warning=True)
def load_data():
    return pd.read_csv("Train.csv", delimiter=";")

data = load_data()

# Streamlit App
def main():
    # Set the title and description of the app
    st.title("Workplace Stress Prediction")
    st.subheader("Our Data")
    st.dataframe(data)
    st.write("This app predicts workplace stress based on user input.")

    with st.form(key="prediction_form"):
        st.write("Enter Your Details")

        # Create input fields for user to enter features
        Age = int(st.number_input("Age", 22))
        Education = int(st.number_input("Education Level", 3))
        AvgDailyHours = int(st.number_input("Average Daily Hours", 8))
        EducationField = st.selectbox("Department", ["Technical Degree", "Medical", "Human Resources", "Marketing", "Other"])
        Department = st.selectbox("Education Field", ["Manufacturing Director", "Sales Representative", "Healthcare Representative", "Research Director", "Manager", "Human Resources", "Sales Executive", "Laboratory Technician", "Research Scientist"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        HasFlexibleTimings = st.selectbox("Has Flexible Timings", ["No", "Yes"])
        IsIndividualContributor = st.selectbox("Is Individual Contributor", ["Yes", "No"])
        JobInvolvement = st.slider("Job Involvement", 1, 4, 2)
        JobRole = st.selectbox("Job Role", ["Manager", "Engineer", "Analyst", "Sales", "HR"])
        JobSatisfaction = st.slider("Job Satisfaction", 1, 5, 3)
        LeavesTaken = int(st.number_input("Leaves Taken", 22))
        MaritalStatus = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
        MicromanagedAtWork = st.slider("Micromanaged at Work", 0, 10, 5)
        MonthlyIncome = int(st.number_input("Monthly Income", 25000))
        NumCompaniesWorked = int(st.number_input("Number of Companies Worked", 2))
        PercentSalaryHike = int(st.number_input("Percent Salary Hike", 45))
        PerformanceRating = st.slider("Performance Rating", 1, 5, 3)
        RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 5, 3)
        RemoteWorkSatisfaction = st.selectbox("Remote Work Satisfaction", ["Medium", "Very High", "High", "Low", "Very Low"])
        SelfMotivationLevel = st.slider("Self Motivation Level", 1, 5, 3)
        TotalWorkingYears = int(st.number_input("Total Working Years", 2))
        TrainingTimesLastYear = int(st.number_input("Training Times Last Year", 3))
        WorkLifeBalance = st.slider("Work-Life Balance", 1, 5, 3)
        WorkLoadLevel = st.selectbox("Work Load Level", ["Low", "High", "Medium"])
        YearsAtCompany = int(st.number_input("Years at Company", 2))
        YearsSinceLastPromotion = int(st.number_input("Years Since Last Promotion", 2))
        YearsWithCurrManager = int(st.number_input("Years with Current Manager", 1))

        if st.form_submit_button("Predict"):
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
            PreProUserData = pd.DataFrame()
            for column in UserData.columns:
                if UserData[column].dtype == object:
                    TempVal = le.fit_transform(UserData[column].astype('category'))
                    PreProUserData[column] = TempVal
                else:
                    PreProUserData[column] = UserData[column]

            st.write("Preprocessed User Data")
            st.dataframe(PreProUserData)

            LRPredicted = LRClassifier.predict(PreProUserData)

            # Display the predictions
            st.subheader("Predictions")
            if LRPredicted == 0:
                st.write("You are a bit STRESSED! Take some assistance..")
            else:
                st.write("You are perfectly alright. Keep Rocking!!")


if __name__ == "__main__":
    main()
