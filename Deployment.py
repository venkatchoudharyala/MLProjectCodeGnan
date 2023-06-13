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

# Define the mapping dictionaries for select boxes
education_field_options = {
    "Technical Degree": 0,
    "Medical": 1,
    "Human Resources": 2,
    "Marketing": 3,
    "Other": 4
}

department_options = {
    "Manufacturing Director": 0,
    "Sales Representative": 1,
    "Healthcare Representative": 2,
    "Research Director": 3,
    "Manager": 4,
    "Human Resources": 5,
    "Sales Executive": 6,
    "Laboratory Technician": 7,
    "Research Scientist": 8
}

gender_options = {
    "Male": 0,
    "Female": 1
}

has_flexible_timings_options = {
    "No": 0,
    "Yes": 1
}

is_individual_contributor_options = {
    "Yes": 0,
    "No": 1
}

marital_status_options = {
    "Divorced": 0,
    "Married": 1,
    "Single": 2
}

job_role_options = {
    "Manager": 0,
    "Engineer": 1,
    "Analyst": 2,
    "Sales": 3,
    "HR": 4
}

remote_work_satisfaction_options = {
    "Medium": 0,
    "Very High": 1,
    "High": 2,
    "Low": 3,
    "Very Low": 4
}

work_load_level_options = {
    "Low": 0,
    "High": 1,
    "Medium": 2
}

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
        EducationField = st.selectbox("Department", list(department_options.keys()))
        Department = st.selectbox("Education Field", list(education_field_options.keys()))
        Gender = st.selectbox("Gender", list(gender_options.keys()))
        HasFlexibleTimings = st.selectbox("Has Flexible Timings", list(has_flexible_timings_options.keys()))
        IsIndividualContributor = st.selectbox("Is Individual Contributor", list(is_individual_contributor_options.keys()))
        JobInvolvement = st.slider("Job Involvement", 1, 4, 2)
        JobRole = st.selectbox("Job Role", list(job_role_options.keys()))
        JobSatisfaction = st.slider("Job Satisfaction", 1, 5, 3)
        LeavesTaken = int(st.number_input("Leaves Taken", 22))
        MaritalStatus = st.selectbox("Marital Status", list(marital_status_options.keys()))
        MicromanagedAtWork = st.slider("Micromanaged at Work", 0, 10, 5)
        MonthlyIncome = int(st.number_input("Monthly Income", 25000))
        NumCompaniesWorked = int(st.number_input("Number of Companies Worked", 2))
        PercentSalaryHike = int(st.number_input("Percent Salary Hike", 45))
        PerformanceRating = st.slider("Performance Rating", 1, 5, 3)
        RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 5, 3)
        RemoteWorkSatisfaction = st.selectbox("Remote Work Satisfaction", list(remote_work_satisfaction_options.keys()))
        SelfMotivationLevel = st.slider("Self Motivation Level", 1, 5, 3)
        TotalWorkingYears = int(st.number_input("Total Working Years", 2))
        TrainingTimesLastYear = int(st.number_input("Training Times Last Year", 3))
        WorkLifeBalance = st.slider("Work-Life Balance", 1, 5, 3)
        WorkLoadLevel = st.selectbox("Work Load Level", list(work_load_level_options.keys()))
        YearsAtCompany = int(st.number_input("Years at Company", 2))
        YearsSinceLastPromotion = int(st.number_input("Years Since Last Promotion", 2))
        YearsWithCurrManager = int(st.number_input("Years with Current Manager", 1))

        if st.form_submit_button("Predict"):
            UserData = pd.DataFrame({
                'EmployeeID': [000000],
                'Age': [Age],
                'AvgDailyHours': [AvgDailyHours],
                'Department': [department_options[Department]],
                'Education': [Education],
                'EducationField': [education_field_options[EducationField]],
                'Gender': [gender_options[Gender]],
                'HasFlexibleTimings': [has_flexible_timings_options[HasFlexibleTimings]],
                'IsIndividualContributor': [is_individual_contributor_options[IsIndividualContributor]],
                'JobInvolvement': [JobInvolvement],
                'JobRole': [job_role_options[JobRole]],
                'JobSatisfaction': [JobSatisfaction],
                'LeavesTaken': [LeavesTaken],
                'MaritalStatus': [marital_status_options[MaritalStatus]],
                'MicromanagedAtWork': [MicromanagedAtWork],
                'MonthlyIncome': [MonthlyIncome],
                'NumCompaniesWorked': [NumCompaniesWorked],
                'PercentSalaryHike': [PercentSalaryHike],
                'PerformanceRating': [PerformanceRating],
                'RelationshipSatisfaction': [RelationshipSatisfaction],
                'RemoteWorkSatistfaction': [remote_work_satisfaction_options[RemoteWorkSatisfaction]],
                'SelfMotivationLevel': [SelfMotivationLevel],
                'TotalWorkingYears': [TotalWorkingYears],
                'TrainingTimesLastYear': [TrainingTimesLastYear],
                'WorkLifeBalance': [WorkLifeBalance],
                'WorkLoadLevel': [work_load_level_options[WorkLoadLevel]],
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
