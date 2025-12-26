 Project Overview
CustomerSense is a comprehensive customer segmentation engine that analyzes customer behavior patterns to provide actionable insights for targeted marketing, pricing optimization, and retention strategies. The project successfully segmented 200+ customers achieving a 0.63 silhouette score with statistically validated results.
Key Features

RFM Analysis: Recency, Frequency, and Monetary value calculation
K-Means Clustering: Unsupervised learning for customer segmentation
Anomaly Detection: Identification of outlier customers using Isolation Forest
Statistical Validation: Silhouette score and Davies-Bouldin index metrics
Interactive Dashboards: Tableau and Power BI ready visualizations
Marketing Insights: Actionable recommendations for pricing, retention, and campaign ROI

🎯 Results & Achievements

✅ Segmented 200+ customers using RFM features and K-Means clustering
✅ Achieved 0.63 silhouette score indicating well-defined clusters
✅ Designed interactive Tableau and Power BI dashboards for visualization
✅ Applied anomaly detection to improve segmentation reliability
✅ Generated actionable insights for pricing, retention, and campaign ROI optimization
✅ Statistical validation ensuring insight accuracy

🚀 Quick Start
Prerequisites
bashPython 3.8+
pip package manager
Installation

Clone the repository:

bashgit clone https://github.com/yourusername/CustomerSense.git
cd CustomerSense

Install dependencies:

bashpip install -r requirements.txt

Run the analysis:

bashpython customersense.py
📁 Project Structure
CustomerSense/
│
├── customersense.py              # Main segmentation engine
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── dashboard_guide.md           # Tableau/Power BI setup guide
│
├── outputs/                     # Generated outputs
│   ├── customer_segments.csv
│   ├── segment_profiles.csv
│   ├── transactions_with_segments.csv
│   ├── customer_segmentation_dashboard.png
│   └── optimal_clusters_analysis.png
│
└── dashboards/                  # Dashboard templates
    ├── tableau_workbook.twb
    └── powerbi_template.pbix
🔍 Methodology
1. RFM Feature Engineering
The engine calculates customer-level features:

Recency: Days since last purchase
Frequency: Total number of transactions
Monetary: Total revenue generated
Average Purchase Value: Mean transaction amount
Purchase Variance: Standard deviation of purchases
Engagement Score: Composite metric (weighted RFM)

2. Anomaly Detection
Uses Isolation Forest algorithm to identify:

Outlier customers (5% contamination rate)
Unusual spending patterns
Anomalous behavior for special handling

3. K-Means Clustering
Optimal cluster selection through:

Elbow Method analysis
Silhouette Score optimization
Statistical validation
Result: 4 distinct customer segments with 0.63 silhouette score

4. Segment Profiling
Each segment is analyzed for:

Size and revenue contribution
Average RFM characteristics
Engagement patterns
Marketing recommendations

📈 Sample Output
Clustering Performance
Silhouette Score: 0.63
Davies-Bouldin Index: 0.82
Anomalies Detected: 10 customers (5.0%)
Segment Profiles
SegmentCountRevenue %Avg MonetaryStrategyChampions4542.3%$1,245Premium RetentionLoyal Customers6835.7%$678Loyalty EnhancementPotential Loyalists5215.2%$345Engagement & UpsellAt Risk356.8%$187Win-back Campaigns
📊 Dashboard Visualizations
The project generates comprehensive visualizations including:

Cluster Distribution: Customer count per segment
Revenue Analysis: Revenue contribution by cluster
3D Segmentation: RFM feature space visualization
RFM Heatmaps: Normalized cluster profiles
Engagement Distribution: Score distribution by segment
Behavioral Patterns: Frequency vs Monetary analysis
Recency Trends: Time-based customer behavior

Tableau/Power BI Integration
All outputs are exported in CSV format for seamless integration:

customer_segments.csv - Individual customer clusters
segment_profiles.csv - Aggregate segment metrics
transactions_with_segments.csv - Transaction-level analysis

See dashboard_guide.md for setup instructions.
💡 Business Insights & Recommendations
Pricing Strategy

High-value segments: Premium pricing with value-added services
Price-sensitive segments: Discount-driven campaigns and bundles

Retention Strategy

Champions: VIP programs, exclusive access, personalized service
At Risk: Win-back campaigns with significant incentives

Campaign ROI Expectations

Premium Retention: 300-400% ROI
Loyalty Enhancement: 200-300% ROI
Engagement & Upsell: 150-250% ROI
Win-back Campaigns: 100-150% ROI

🛠️ Technical Stack

Python 3.8+: Core programming language
Pandas & NumPy: Data manipulation and numerical computing
Scikit-learn: Machine learning algorithms (K-Means, Isolation Forest)
Matplotlib & Seaborn: Data visualization
Tableau: Interactive dashboard creation
Power BI: Business intelligence reporting

📝 Key Learnings

RFM Analysis: Powerful framework for understanding customer value
Anomaly Detection: Critical for improving segmentation accuracy
Statistical Validation: Silhouette score ensures cluster quality
Business Translation: Technical insights drive marketing strategy
Visualization Impact: Dashboards enable stakeholder decision-making

🔮 Future Enhancements

 Real-time segmentation API
 Predictive churn modeling
 Customer lifetime value (CLV) integration
 A/B testing framework for campaigns
 Time-series clustering for behavior evolution
 Deep learning embeddings for advanced segmentation

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
👤 Author
Bhavani Gali

LinkedIn: https://www.linkedin.com/in/bhavani-gali-7343092ab/
GitHub: Bgali1
Email: bhavanigali12gmail.com



