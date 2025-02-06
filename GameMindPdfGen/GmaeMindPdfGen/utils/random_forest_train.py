import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
from fpdf import FPDF
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import os
import numpy as np
import io


def random_forest_train(data_fill):
    user_streaks = data_fill.groupby("User")["Streak"].agg(["first", "last"])
    user_streaks["Improved"] = (user_streaks["last"] > user_streaks["first"]).astype(
        int
    )
    data_fill = data_fill.merge(
        user_streaks[["Improved"]], left_on="User", right_index=True
    )
    label_Data = LabelEncoder()
    data_fill["Level_Encoded"] = label_Data.fit_transform(data_fill["Level"])
    data_fill["Session_Encoded"] = label_Data.fit_transform(data_fill["Session"])
    X_features = data_fill[["Level_Encoded", "Session_Encoded", "Age", "Time"]]
    y_Target = data_fill["Improved"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_Target, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        X_features,
        y_Target,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 0.9, 60),
        random_state=42,
    )
    Randomtrain_mean = np.mean(train_scores, axis=1)
    Randomtrain_std = np.std(train_scores, axis=1)
    Randomvalidation_mean = np.mean(validation_scores, axis=1)
    Randomvalidation_std = np.std(validation_scores, axis=1)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    cv_scores = cross_val_score(model, X_features, y_Target, cv=4, scoring="accuracy")
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")

    model = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [40, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [1, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="accuracy",
        verbose=2,
    )
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_}")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "random_forest_model.pkl")


def load_model(data_fill):
    model = joblib.load("random_forest_model.pkl")
    # Prepare your data
    X_features = data_fill[["Level_Encoded", "Session_Encoded", "Age", "Time"]]
    y_Target = data_fill["Improved"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_Target, test_size=0.3, random_state=42
    )

    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, "random_forest_model.pkl")

   

    # Make predictions and evaluate the model
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def generate_cognitive_report(username,data_fill):
    user_streaks = data_fill.groupby("User")["Streak"].agg(["first", "last"])
    user_streaks["Improved"] = (user_streaks["last"] > user_streaks["first"]).astype(
        int
    )
    data_fill = data_fill.merge(
        user_streaks[["Improved"]], left_on="User", right_index=True
    )
    label_Data = LabelEncoder()
    data_fill["Level_Encoded"] = label_Data.fit_transform(data_fill["Level"])
    data_fill["Session_Encoded"] = label_Data.fit_transform(data_fill["Session"])
    X_features = data_fill[['Level_Encoded', 'Session_Encoded', 'Age', 'Time']]
    y_Target = data_fill['Improved']
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_Target, test_size=0.3, random_state=42)
    # Load the trained model
    model = joblib.load('random_forest_model.pkl')

    # Extract data for the specific user
    user_data = data_fill[data_fill['User'] == username]
    print(user_data)
    # Prepare the user's data in the format expected by the model
    X_user = user_data[['Level_Encoded', 'Session_Encoded', 'Age', 'Time']]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_user_scaled = scaler.transform(X_user) 
    
    # Predict whether there is cognitive improvement or decline
    user_prediction = model.predict(X_user_scaled)

    user_probabilities = model.predict_proba(X_user_scaled)

    #  Create a new column for improvement probability in the user_data DataFrame 
    user_data['Improvement_Probability'] = user_probabilities[:, 1] 

    # Create a PDF report
    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # Set title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt=f"Cognitive Performance Report for {username}", ln=True, align='C')

    # Add some basic information
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, txt=f"Total Sessions: {len(user_data)}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Improvement Probability: {user_probabilities[-1][1]:.2f}", ln=True, align='L')

    # Analyze the result
    if user_prediction[-1] == 1:
        pdf.cell(200, 10, txt=f"Insight: {username} is likely showing cognitive improvement.", ln=True, align='L')
    else:
        pdf.cell(200, 10, txt=f"Insight: {username} might not be showing improvement, or there could be cognitive decline.", ln=True, align='L')

    pdf.cell(200, 10, txt="Detailed insights and plots are provided below.", ln=True, align='L')

    # Generate plots and add them to the PDF
    # Plot 1: Streak over Sessions
    plt.figure(figsize=(5, 4))
    plt.plot(user_data['Session'], user_data['Streak'], marker='o', color='blue')
    plt.xlabel('Session')
    plt.ylabel('Streak')
    plt.title(f'Streak over Sessions for {username}')
    plt.savefig('plot1.png')

    # Plot 2: Time Spent per Session
    plt.figure(figsize=(5, 4))
    plt.plot(user_data['Session'], user_data['Time'], marker='o', color='red')
    plt.xlabel('Session')
    plt.ylabel('Time')
    plt.title(f'Time Spent per Session for {username}')
    plt.savefig('plot2.png')

    # Plot 3: Improvement Probability over Sessions
    plt.figure(figsize=(5, 4))
    plt.plot(user_data['Session'], user_data['Improvement_Probability'], marker='o', color='green')
    plt.xlabel('Session')
    plt.ylabel('Improvement Probability')
    plt.title(f'Improvement Probability over Sessions for {username}')
    plt.savefig('plot3.png')

    # Plot 4: Level Progression over Sessions
    plt.figure(figsize=(5, 4))
    plt.plot(user_data['Session'], user_data['Level_Encoded'], marker='o', color='purple')
    plt.xlabel('Session')
    plt.ylabel('Level (Encoded)')
    plt.title(f'Level Progression over Sessions for {username}')
    plt.savefig('plot4.png')

    # Insert plots into the PDF
    pdf.image('plot1.png', x=10, y=60, w=90)
    pdf.image('plot2.png', x=10, y=140, w=90)
    pdf.image('plot3.png', x=110, y=60, w=90)
    pdf.image('plot4.png', x=110, y=140, w=90)

    # Save the PDF
    # pdf_filename = f"{username}_Cognitive_Report.pdf"
    
    pdf_content = pdf.output(dest='S').encode('latin1')
    pdf_output = io.BytesIO(pdf_content)
    # pdf.output(pdf_output)
    pdf_output.seek(0)
    # Clean up temporary plot files
    os.remove('plot1.png')
    os.remove('plot2.png')
    os.remove('plot3.png')
    os.remove('plot4.png')

    # Download the PDF file if running in Colab
    return pdf_output
