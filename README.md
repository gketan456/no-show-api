🏥 Healthcare Appointment No-Show Prediction System
1️⃣ Problem Statement

Healthcare providers lose significant revenue and operational efficiency when patients miss scheduled appointments (“no-shows”).

Missed appointments result in:

Wasted clinician time

Increased operational costs

Reduced patient access

Lower clinic utilization

The goal of this project is to build an end-to-end machine learning system that predicts the probability that a patient will miss a scheduled appointment, allowing hospitals to proactively intervene.

If the predicted risk exceeds a defined threshold (0.7), the patient is flagged for targeted intervention.

2️⃣ Dataset Description

Source: Kaggle – Medical Appointment No Shows Dataset

Records: 110,527 appointments

Target Variable: No-show (Yes / No)

Available Features

Patient demographics (Age, Gender)

Social support indicator (Scholarship)

Chronic condition flags (Hypertension, Diabetes, Alcoholism)

Disability indicator (Handcap)

SMS reminder status

Appointment scheduling timestamps

Neighborhood information

The dataset reflects real-world hospital scheduling data and simulates operational healthcare analytics.

3️⃣ Feature Engineering Logic

To simulate real hospital operations, additional meaningful features were engineered:

⏳ Temporal Features

lead_time_days = AppointmentDay − ScheduledDay
(Captures how far in advance the appointment was booked)

appointment_dayofweek

appointment_weekend

👥 Demographic Features

Age grouped into categories

One-hot encoded age groups

📍 Geographic Feature

neighbourhood_freq
Frequency encoding of neighborhood (avoids high-dimensional one-hot encoding)

📩 Behavioral Indicators

SMS reminder effect (SMS_received)

Chronic condition indicators

Scholarship status

These features were designed to reflect realistic healthcare behavior patterns rather than relying solely on raw fields.

4️⃣ Model Results
Model Used

Logistic Regression

class_weight="balanced" to handle class imbalance

The dataset is imbalanced (~20% no-shows), so class weighting was applied to improve recall for the minority class.

Evaluation Metrics (Test Set)

AUC-ROC: ~0.66

Accuracy: ~0.66

Recall (No-show class): ~0.57

Precision (No-show class): ~0.31

The model provides moderate predictive power while remaining interpretable — important in healthcare environments.

5️⃣ API Usage Instructions

The trained model is deployed as a FastAPI application and containerized using Docker.

Available Endpoints
Health Check
GET /health

Readiness Check
GET /ready

Prediction Endpoint
POST /v1/predict

Example Request
{
  "Gender": "F",
  "Age": 62,
  "Neighbourhood": "JARDIM DA PENHA",
  "ScheduledDay": "2016-04-29T18:38:08Z",
  "AppointmentDay": "2016-04-29",
  "Scholarship": 0,
  "Hipertension": 1,
  "Diabetes": 0,
  "Alcoholism": 0,
  "Handcap": 0,
  "SMS_received": 0
}

Example Response
{
  "no_show_probability": 0.33,
  "risk_flag": false,
  "threshold": 0.7,
  "recommended_action": "STANDARD_REMINDER",
  "model_version": "v1.0.0"
}

Running Locally with Docker
docker build -t no-show-api .
docker run -p 8000:8000 no-show-api


Open Swagger UI:

http://127.0.0.1:8000/docs

6️⃣ Business Impact Explanation

Instead of contacting every patient, hospitals can use this system to:

Identify high-risk no-show patients

Apply targeted reminder strategies

Improve clinic utilization

Reduce financial loss from missed appointments

Using a risk threshold of 0.7:

Only ~6–7% of patients are flagged

Enables cost-effective, targeted intervention

This project simulates a real healthcare operations ML deployment pipeline — from raw data processing to production-ready API deployment.

🛠 Tech Stack

Python

Pandas

Scikit-learn

FastAPI

Docker

Render (Cloud Deployment)
