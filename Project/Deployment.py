import streamlit as st
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay
import plotly.express as px

st.set_page_config(page_title="WorkPlace Health", 
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded")

st.set_option('deprecation.showPyplotGlobalUse', False)

st.sidebar.title("Model Where Abouts..")

# Load the trained models
DTClassifier = pickle.load(open('PickleFiles/DecisionTree.pkl', 'rb'))
RFClassifier = pickle.load(open('PickleFiles/RandomForest.pkl', 'rb'))
SVM = pickle.load(open('PickleFiles/SVM.pkl', 'rb'))
KNN = pickle.load(open('PickleFiles/KNN.pkl', 'rb'))
LRClassifier = pickle.load(open('PickleFiles/LogisticRegression.pkl', 'rb'))

# Load the data
def LoadData():
    return pd.read_csv("Train.csv", delimiter=";")

data = LoadData()

# Define the mapping dictionaries for select boxes
Edu = {
    "UnderGraduate": 3,
    "+10": 1,
    "+12": 2,
    "PostGraduate": 4,
    "ResearchGraduate": 5
}

EduFielOpt = {
    "Technical Degree": 0,
    "Medical": 1,
    "Human Resources": 2,
    "Marketing": 3,
    "Other": 4
}

DptOpt = {
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

GenOpt = {
    "Male": 0,
    "Female": 1
}

FlexTime = {
    "No": 0,
    "Yes": 1
}

ContrInd = {
    "Yes": 0,
    "No": 1
}

Married = {
    "Divorced": 0,
    "Married": 1,
    "Single": 2
}

JobRoleOpt = {
    "Manager": 0,
    "Engineer": 1,
    "Analyst": 2,
    "Sales": 3,
    "HR": 4
}

RemWork = {
    "Medium": 0,
    "Very High": 1,
    "High": 2,
    "Low": 3,
    "Very Low": 4
}

WorkLoad = {
    "Low": 0,
    "High": 1,
    "Medium": 2
}

# Streamlit App
def main():
    # Set the title and description of the app
    st.title("Workplace Stress Predictor")
        
    Frame1, Frame2, Frame3 = st.tabs(["InteractiveVisual", "Predict Yours", "About Our Data"])
        
    with Frame2:
        with st.form(key="prediction_form"):
            st.write("Enter Your Details")

            # Create input fields for user to enter features
            Age = int(st.number_input("Age", value = 22))
            Education = st.selectbox("Education Level", list(Edu.keys()))
            AvgDailyHours = int(st.number_input("Average Daily Hours", value = 8))
            Department = st.selectbox("Department", list(DptOpt.keys()))
            EducationField = st.selectbox("Education Field", list(EduFielOpt.keys()))
            Gender = st.selectbox("Gender", list(GenOpt.keys()))
            HasFlexibleTimings = st.selectbox("Has Flexible Timings", list(FlexTime.keys()))
            IsIndividualContributor = st.selectbox("Is Individual Contributor", list(ContrInd.keys()))
            JobInvolvement = st.slider("Job Involvement", 1, 4, 2)
            JobRole = st.selectbox("Job Role", list(JobRoleOpt.keys()))
            JobSatisfaction = st.slider("Job Satisfaction", 1, 4, 3)
            LeavesTaken = int(st.number_input("Leaves Taken", value = 22))
            MaritalStatus = st.selectbox("Marital Status", list(Married.keys()))
            MicromanagedAtWork = st.slider("Micromanaged at Work", 1, 5, 3)
            MonthlyIncome = int(st.number_input("Monthly Income", value = 25000))
            NumCompaniesWorked = int(st.number_input("Number of Companies Worked",value = 2))
            PercentSalaryHike = int(st.number_input("Percent Salary Hike", value = 45))
            PerformanceRating = st.slider("Performance Rating", 1, 4, 3)
            RelationshipSatisfaction = st.slider("Relationship Satisfaction", 1, 4, 2)
            RemoteWorkSatisfaction = st.selectbox("Remote Work Satisfaction", list(RemWork.keys()))
            SelfMotivationLevel = st.slider("Self Motivation Level", 1, 4, 3)
            TotalWorkingYears = int(st.number_input("Total Working Years", value = 2))
            TrainingTimesLastYear = int(st.number_input("Training Times Last Year", value = 3))
            WorkLifeBalance = st.slider("Work-Life Balance", 1, 4, 3)
            WorkLoadLevel = st.selectbox("Work Load Level", list(WorkLoad.keys()))
            YearsAtCompany = int(st.number_input("Years at Company",value = 2))
            YearsSinceLastPromotion = int(st.number_input("Years Since Last Promotion", value = 2))
            YearsWithCurrManager = int(st.number_input("Years with Current Manager", value = 1))
            if st.form_submit_button("Predict"):
                UserData = pd.DataFrame({
                    'Age': [Age],
                    'AvgDailyHours': [AvgDailyHours],
                    'Department': [DptOpt[Department]],
                    'Education': [Edu[Education]],
                    'EducationField': [EduFielOpt[EducationField]],
                    'Gender': [GenOpt[Gender]],
                    'HasFlexibleTimings': [FlexTime[HasFlexibleTimings]],
                    'IsIndividualContributor': [ContrInd[IsIndividualContributor]],
                    'JobInvolvement': [JobInvolvement],
                    'JobRole': [JobRoleOpt[JobRole]],
                    'JobSatisfaction': [JobSatisfaction],
                    'LeavesTaken': [LeavesTaken],
                    'MaritalStatus': [Married[MaritalStatus]],
                    'MicromanagedAtWork': [MicromanagedAtWork],
                    'MonthlyIncome': [MonthlyIncome],
                    'NumCompaniesWorked': [NumCompaniesWorked],
                    'PercentSalaryHike': [PercentSalaryHike],
                    'PerformanceRating': [PerformanceRating],
                    'RelationshipSatisfaction': [RelationshipSatisfaction],
                    'RemoteWorkSatistfaction': [RemWork[RemoteWorkSatisfaction]],
                    'SelfMotivationLevel': [SelfMotivationLevel],
                    'TotalWorkingYears': [TotalWorkingYears],
                    'TrainingTimesLastYear': [TrainingTimesLastYear],
                    'WorkLifeBalance': [WorkLifeBalance],
                    'WorkLoadLevel': [WorkLoad[WorkLoadLevel]],
                    'YearsAtCompany': [YearsAtCompany],
                    'YearsSinceLastPromotion': [YearsSinceLastPromotion],
                    'YearsWithCurrManager': [YearsWithCurrManager]
                })

                le = LabelEncoder()
                PreProUserData = pd.DataFrame()
                for column in UserData.columns:
                    if UserData[column].dtype == object:
                        TempVal = le.fit_transform(UserData[column].astype('category'))
                        PreProUserData[column] = TempVal
                    else:
                        PreProUserData[column] = UserData[column]
                        
                #st.write("Your Details after Pre-processing")

                #st.dataframe(PreProUserData)

                LRPredicted = RFClassifier.predict(PreProUserData)

                # Display the predictions
                with st.spinner("We're almost there.."):
                    time.sleep(3)
                    st.subheader("Predictions")
                    if LRPredicted == 1:
                        st.warning("You are a bit STRESSED! Take some assistance..")
                    else:
                        st.balloons()
                        st.write("You are perfectly alright. Keep Rocking!!") 

    with Frame3:
        st.subheader("Our Data")
        st.dataframe(data)

        ModifiedData = data.copy()
        PreProModifiedData = pd.DataFrame()
        le = LabelEncoder()
        for column in ModifiedData.columns:
            if ModifiedData[column].dtype == object:
                TempVal = le.fit_transform(ModifiedData[column].astype('category'))
                PreProModifiedData[column] = TempVal
            else:
                PreProModifiedData[column] = ModifiedData[column]
        PreProModifiedData.drop("EmployeeID", axis = 1, inplace = True)
        PreProModifiedData.dropna(inplace = True)

        st.subheader("Interactive Visualization")
        st.markdown("---")
        # Interactive Plotly graph 1: Scatter plot
        st.subheader("ScatterPlot on (Age, MonthlyIncome) and JobRole as color")
        scatter_plot = px.scatter(PreProModifiedData, x="Age", y="MonthlyIncome", color="JobRole")
        st.plotly_chart(scatter_plot)
        st.markdown("----")

        st.subheader("BarChart on (EducationField, TotalWorkingYears) and Department as color")
        # Interactive Plotly graph 2: Bar chart
        bar_chart = px.bar(PreProModifiedData, x="EducationField", y="TotalWorkingYears", color="Department")
        st.plotly_chart(bar_chart)
        st.markdown("---")

        st.subheader("LinePlot on (YearsAtCompany, JobSatisfaction) and MaritalStatus as color")
        # Interactive Plotly graph 3: Line plot
        line_plot = px.line(PreProModifiedData, x="YearsAtCompany", y="JobSatisfaction", color="MaritalStatus")
        st.plotly_chart(line_plot)
        st.markdown("---")

        st.subheader("PieChart on Gender")
        # Interactive Plotly graph 4: Pie chart
        pie_chart = px.pie(PreProModifiedData, names="Gender")
        st.plotly_chart(pie_chart)
        st.markdown("---")

        st.subheader("3D-ScatterPlot on (YearsWithCurrManager, YearsSinceLastPromotion, PercentSalaryHike) and PerformanceRating as color")
        # Interactive Plotly graph 5: 3D scatter plot
        scatter_3d = px.scatter_3d(PreProModifiedData, x="YearsWithCurrManager", y="YearsSinceLastPromotion", z="PercentSalaryHike", color="PerformanceRating")
        st.plotly_chart(scatter_3d)
        st.markdown("---")

    with open("Project/ViewCount.txt", "r") as file:
        ViewCount = file.read()
        
    ViewCount += 1
    
    with open("Project/ViewCount.txt", "w") as file:
        file.write("ViewCount")
        
    Algo = st.sidebar.selectbox("Select Algo", ("RandomForest", "DecisionTree", "SVM", "LogisticRegression", "KNN"))
    Metrics = st.sidebar.multiselect("What Metrics to plot?", ("ConfusionMatrix","RocCurve","PrecisionRecallCurve"))
    ExtMet = st.sidebar.multiselect("Statistical Metrics", ("AccuracyScore", "Precision"))
     
    y_test = pickle.load(open('PickleFiles/YTest.pkl', 'rb'))
    x_test = pickle.load(open('PickleFiles/XTest.pkl', 'rb'))

    if Algo == "RandomForest":
        y_pred = RFClassifier.predict(x_test)
        Model = RFClassifier
    elif Algo == "DecisionTree":
        y_pred = DTClassifier.predict(x_test)
        Model = DTClassifier
    elif Algo == "SVM":
        y_pred = SVM.predict(x_test)
        Model = SVM
    elif Algo == "LogisticRegression":
        y_pred = LRClassifier.predict(x_test)
        Model = LRClassifier
    elif Algo == "KNN":
        y_pred = KNN.predict(x_test)
        Model = KNN
        
    def PlotMetrics(Metrics):
        if "ConfusionMatrix" in Metrics:
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(Model,x_test,y_test)
            st.pyplot()
        if "RocCurve" in Metrics:
            st.subheader("ROC Curve")
            RocCurveDisplay.from_estimator(Model,x_test,y_test)
            st.pyplot()
        if "PrecisionRecallCurve" in Metrics:
            st.subheader("Precision-Recall Curve")
            PrecisionRecallDisplay.from_estimator(Model,x_test,y_test)
            st.pyplot()
            
    def ExtrMetrics(ExtMet):  
        if "AccuracyScore" in ExtMet:
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader("Accuracy:")
            st.write(accuracy.round(4))
        if "Precision" in ExtMet:
            # Calculate precision
            precision = precision_score(y_test, y_pred)
            st.subheader("Precision:")
            st.write(precision.round(4))
        
    if st.sidebar.button("Visualize", key = "Visualize"):
        with Frame1:
           PlotMetrics(Metrics)
           ExtrMetrics(ExtMet)  
           st.markdown("---")
        
    st.sidebar.write("Number of Views: ",ViewCount)
            
    with Frame1:
        st.subheader("Project Abstract")
        st.markdown("---")
        st.write("New phase of competitive market!!!!")
        st.write("It's primary motive is to fulfill the requirements according to the Deadlines. Which inturn curtailing the personal and recreational time of the Employee. This may adversly affect Employee's Mental health. That may even deplete his performance.")
        st.write("As we all know Prevention is better than cure, this particular application will be helping individuals and firms to track the stress levels of the Employees such that, they may be assissted time to time. This will also boost their performance and productivity. ")
        st.subheader("Our Data model has few specific Features like:")
        st.write("1) Job Satisfaction")
        st.write("2) Has Flexibile Timings")
        st.write("3) Micro-managed at work")
        st.write("4) Percent salary Hike")
        st.write("5) Self Motivation level")
        st.write("6) Work life Balance etc.,")
        st.write("We can Predict an all-round mental state of the Employee..")
        st.subheader("Tech Stack")
        st.write("1) Libraries : plotly, sklearn, matplotlib, time, pandas")
        st.write("2) Programmig Language : Python3")
        st.write("3) FrameWork : Streamlit")
        
if __name__ == "__main__":
    main()
