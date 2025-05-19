# titanic-disaster-ml
Machine learning project using clustering and classification to analyze and predict Titanic passenger survival.

# Titanic Disaster Prediction Project 🚢  
This project applies both unsupervised and supervised machine learning techniques to analyze and predict survival outcomes from the Titanic disaster. It was created using Python as part of a Machine Learning course (CS379).

---

## 📁 Dataset  
**Source**: titanicdisaster_dataset.xlsx (custom/prepared version based on the Titanic passenger manifest)  
The dataset includes:  
- `Pclass`: Passenger class (1st, 2nd, 3rd)  
- `Sex`: Gender of the passenger  
- `Age`: Age of the passenger  
- `Fare`: Ticket fare paid  
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)  
- `Survived`: Survival status (0 = No, 1 = Yes)

---

## 🔍 Unsupervised Learning  
**Algorithm**: KMeans Clustering (from sklearn.cluster)  
**Goal**: Explore and group passengers into clusters based on features such as age, class, and fare to discover underlying structure or groupings in the data.

---

## 🧠 Supervised Learning  
**Algorithm**: Decision Tree Classifier (from sklearn.tree)  
**Goal**: Predict whether a passenger survived the Titanic disaster using features like class, age, gender, and fare.

---

## 🔧 Preprocessing Steps  
- Converted categorical variables (e.g., Sex, Embarked) to numeric using label encoding or one-hot encoding  
- Handled missing values by imputing median (for Age) or most frequent value (for Embarked)  
- Standardized numerical features where necessary  
- Split the dataset into training and test sets for the supervised model

---

## 📈 Results  
Evaluated using:  
- **Accuracy Score**  
- **Confusion Matrix**  
- **Cluster Analysis Visualizations** (for unsupervised part)

---

## ▶️ How to Run  
Clone the repo:  
```bash
git clone https://github.com/yourusername/titanic-disaster-ml.git  
cd titanic-disaster-ml
