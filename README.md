# 🧠 Customer Segmentation for Personalized Marketing

This project applies **K-Means Clustering** to segment mall customers based on their age, income, and spending behavior. The goal is to help businesses better target customers through personalized marketing — a task commonly undertaken by data teams at companies like JPMorgan.

---

## 📂 Dataset

- 📊 **Total Records**: 200 customers  
- 📁 **Features Used**:
  - `Gender`
  - `Age`
  - `Annual Income (k$)`
  - `Spending Score (1–100)`

Dataset Source: Simulated mall customer data.

---

## 🔧 Tools Used

- 🐍 Python (Pandas, Scikit-learn, Seaborn, Matplotlib)
- 📓 Jupyter Notebook
- 📊 Tableau (for dashboards)
- 💻 Git + GitHub

---

## 🧪 Project Workflow

### 1️⃣ Data Preprocessing
- Checked for missing values (none found)
- Label encoded the `Gender` column
- Applied feature scaling for numerical features

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized income vs spending, age vs spending, and cluster distributions
- Used boxplots and pairplots to identify cluster traits

### 3️⃣ K-Means Clustering
- Used the Elbow Method to determine optimal `k = 5`
- Trained K-Means model on scaled features
- Assigned cluster labels to all customer records

### 4️⃣ Tableau Dashboard
- Created an interactive dashboard to visualize clusters and spending patterns
- Enabled dynamic filters to explore cluster-based insights

---

## 📈 Key Insights

- Found clear separation between high-income low-spending and low-income high-spending groups
- Identified ideal customer groups for targeted marketing
- Visualized 5 unique customer personas/clusters

---

## 📄 Files Included

| File Name                               | Description                                 |
|----------------------------------------|---------------------------------------------|
| `customer_segmentation.ipynb`          | Python notebook with code and EDA           |
| `customer_data.csv`                    | Cleaned dataset used for analysis           |
| `Customer_Segmentation_Executive_Summary.pdf` | One-page project summary               |
| `Tableau_Dashboard_Screenshot.png`     | Image preview of Tableau dashboard          |

---

## 📊 Tableau Dashboard

If available, view the interactive dashboard here:  
🔗 [Insert your Tableau Public link once published]

---

## 👩🏻‍💻 Author

**Bhavani Gali**  
Data Science Major, Arizona State University  
📧 bhavanigali@email.com  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/yourprofile) *(optional)*

---

## ⭐️ If you found this helpful...

Give this repo a ⭐️ and check out more of my work at:  
🔗 [https://github.com/your-username](https://github.com/your-username)


