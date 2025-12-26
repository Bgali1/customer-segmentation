{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 .AppleSystemUIFontMonospaced-Regular;\f1\fnil\fcharset0 .AppleSystemUIFontMonospaced-RegularItalic;}
{\colortbl;\red255\green255\blue255;\red136\green185\blue102;\red36\green36\blue35;\red155\green162\blue177;
\red184\green93\blue213;\red74\green80\blue93;\red197\green136\blue83;\red81\green157\blue235;}
{\*\expandedcolortbl;;\cssrgb\c59608\c76471\c47451;\cssrgb\c18824\c18824\c18039;\cssrgb\c67059\c69804\c74902;
\cssrgb\c77647\c47059\c86667;\cssrgb\c36078\c38824\c43922;\cssrgb\c81961\c60392\c40000;\cssrgb\c38039\c68627\c93725;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4 CustomerSense - Customer Segmentation Engine\
JPMorgan Challenge Project\
Author: Bhavani Gali\
Description: Advanced customer segmentation using RFM analysis, K-Means clustering,\
             anomaly detection, and statistical validation for targeted marketing.\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2 """\cf4 \strokec4 \
\
\pard\pardeftab720\partightenfactor0
\cf5 \strokec5 import\cf4 \strokec4  pandas \cf5 \strokec5 as\cf4 \strokec4  pd\
\cf5 \strokec5 import\cf4 \strokec4  numpy \cf5 \strokec5 as\cf4 \strokec4  np\
\cf5 \strokec5 from\cf4 \strokec4  datetime \cf5 \strokec5 import\cf4 \strokec4  datetime, timedelta\
\cf5 \strokec5 from\cf4 \strokec4  sklearn.preprocessing \cf5 \strokec5 import\cf4 \strokec4  StandardScaler\
\cf5 \strokec5 from\cf4 \strokec4  sklearn.cluster \cf5 \strokec5 import\cf4 \strokec4  KMeans\
\cf5 \strokec5 from\cf4 \strokec4  sklearn.metrics \cf5 \strokec5 import\cf4 \strokec4  silhouette_score, davies_bouldin_score\
\cf5 \strokec5 from\cf4 \strokec4  sklearn.ensemble \cf5 \strokec5 import\cf4 \strokec4  IsolationForest\
\cf5 \strokec5 import\cf4 \strokec4  matplotlib.pyplot \cf5 \strokec5 as\cf4 \strokec4  plt\
\cf5 \strokec5 import\cf4 \strokec4  seaborn \cf5 \strokec5 as\cf4 \strokec4  sns\
\cf5 \strokec5 import\cf4 \strokec4  warnings\
warnings.filterwarnings(\cf2 \strokec2 'ignore'\cf4 \strokec4 )\
\
\pard\pardeftab720\partightenfactor0

\f1\i \cf6 \strokec6 # Set random seed for reproducibility
\f0\i0 \cf4 \strokec4 \
np.random.seed(\cf7 \strokec7 42\cf4 \strokec4 )\
\
\pard\pardeftab720\partightenfactor0
\cf5 \strokec5 class\cf4 \strokec4  \cf7 \strokec7 CustomerSegmentationEngine\cf4 \strokec4 :\
    \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4     Advanced Customer Segmentation Engine using RFM Analysis and K-Means Clustering\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2     """\cf4 \strokec4 \
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf8 \strokec8 __init__\cf4 \strokec4 (self):\
        self.scaler \cf8 \strokec8 =\cf4 \strokec4  StandardScaler()\
        self.kmeans_model \cf8 \strokec8 =\cf4 \strokec4  \cf7 \strokec7 None\cf4 \strokec4 \
        self.customer_data \cf8 \strokec8 =\cf4 \strokec4  \cf7 \strokec7 None\cf4 \strokec4 \
        self.rfm_data \cf8 \strokec8 =\cf4 \strokec4  \cf7 \strokec7 None\cf4 \strokec4 \
        self.segmented_data \cf8 \strokec8 =\cf4 \strokec4  \cf7 \strokec7 None\cf4 \strokec4 \
        self.anomaly_detector \cf8 \strokec8 =\cf4 \strokec4  IsolationForest(contamination\cf8 \strokec8 =\cf7 \strokec7 0.05\cf4 \strokec4 , random_state\cf8 \strokec8 =\cf7 \strokec7 42\cf4 \strokec4 )\
        \
    \cf5 \strokec5 def\cf4 \strokec4  \cf8 \strokec8 generate_synthetic_data\cf4 \strokec4 (self, n_customers\cf8 \strokec8 =\cf7 \strokec7 200\cf4 \strokec4 ):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Generate synthetic customer transaction data for demonstration\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "Generating synthetic customer transaction data..."\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Generate customer IDs
\f0\i0 \cf4 \strokec4 \
        customer_ids \cf8 \strokec8 =\cf4 \strokec4  [\cf2 \strokec2 f'CUST_\cf4 \strokec4 \{\cf2 \strokec2 str\cf4 \strokec4 (i).zfill(\cf7 \strokec7 4\cf4 \strokec4 )\}\cf2 \strokec2 '\cf4 \strokec4  \cf5 \strokec5 for\cf4 \strokec4  i \cf5 \strokec5 in\cf4 \strokec4  \cf2 \strokec2 range\cf4 \strokec4 (\cf7 \strokec7 1\cf4 \strokec4 , n_customers \cf8 \strokec8 +\cf4 \strokec4  \cf7 \strokec7 1\cf4 \strokec4 )]\
        \
        
\f1\i \cf6 \strokec6 # Generate transaction data with realistic patterns
\f0\i0 \cf4 \strokec4 \
        transactions \cf8 \strokec8 =\cf4 \strokec4  []\
        \
        \cf5 \strokec5 for\cf4 \strokec4  cust_id \cf5 \strokec5 in\cf4 \strokec4  customer_ids:\
            
\f1\i \cf6 \strokec6 # Number of transactions per customer (varied distribution)
\f0\i0 \cf4 \strokec4 \
            n_transactions \cf8 \strokec8 =\cf4 \strokec4  np.random.choice([\cf7 \strokec7 1\cf4 \strokec4 , \cf7 \strokec7 2\cf4 \strokec4 , \cf7 \strokec7 3\cf4 \strokec4 , \cf7 \strokec7 5\cf4 \strokec4 , \cf7 \strokec7 8\cf4 \strokec4 , \cf7 \strokec7 12\cf4 \strokec4 , \cf7 \strokec7 15\cf4 \strokec4 , \cf7 \strokec7 20\cf4 \strokec4 , \cf7 \strokec7 25\cf4 \strokec4 ], \
                                             p\cf8 \strokec8 =\cf4 \strokec4 [\cf7 \strokec7 0.15\cf4 \strokec4 , \cf7 \strokec7 0.15\cf4 \strokec4 , \cf7 \strokec7 0.15\cf4 \strokec4 , \cf7 \strokec7 0.15\cf4 \strokec4 , \cf7 \strokec7 0.15\cf4 \strokec4 , \cf7 \strokec7 0.1\cf4 \strokec4 , \cf7 \strokec7 0.08\cf4 \strokec4 , \cf7 \strokec7 0.05\cf4 \strokec4 , \cf7 \strokec7 0.02\cf4 \strokec4 ])\
            \
            
\f1\i \cf6 \strokec6 # Generate transaction dates over past 365 days
\f0\i0 \cf4 \strokec4 \
            \cf5 \strokec5 for\cf4 \strokec4  _ \cf5 \strokec5 in\cf4 \strokec4  \cf2 \strokec2 range\cf4 \strokec4 (n_transactions):\
                days_ago \cf8 \strokec8 =\cf4 \strokec4  np.random.randint(\cf7 \strokec7 1\cf4 \strokec4 , \cf7 \strokec7 366\cf4 \strokec4 )\
                transaction_date \cf8 \strokec8 =\cf4 \strokec4  datetime.now() \cf8 \strokec8 -\cf4 \strokec4  timedelta(days\cf8 \strokec8 =\cf4 \strokec4 days_ago)\
                \
                
\f1\i \cf6 \strokec6 # Generate monetary value with different customer segments
\f0\i0 \cf4 \strokec4 \
                \cf5 \strokec5 if\cf4 \strokec4  np.random.random() \cf8 \strokec8 <\cf4 \strokec4  \cf7 \strokec7 0.2\cf4 \strokec4 :  
\f1\i \cf6 \strokec6 # High-value customers
\f0\i0 \cf4 \strokec4 \
                    amount \cf8 \strokec8 =\cf4 \strokec4  np.random.uniform(\cf7 \strokec7 500\cf4 \strokec4 , \cf7 \strokec7 2000\cf4 \strokec4 )\
                \cf5 \strokec5 elif\cf4 \strokec4  np.random.random() \cf8 \strokec8 <\cf4 \strokec4  \cf7 \strokec7 0.5\cf4 \strokec4 :  
\f1\i \cf6 \strokec6 # Medium-value customers
\f0\i0 \cf4 \strokec4 \
                    amount \cf8 \strokec8 =\cf4 \strokec4  np.random.uniform(\cf7 \strokec7 100\cf4 \strokec4 , \cf7 \strokec7 500\cf4 \strokec4 )\
                \cf5 \strokec5 else\cf4 \strokec4 :  
\f1\i \cf6 \strokec6 # Low-value customers
\f0\i0 \cf4 \strokec4 \
                    amount \cf8 \strokec8 =\cf4 \strokec4  np.random.uniform(\cf7 \strokec7 10\cf4 \strokec4 , \cf7 \strokec7 100\cf4 \strokec4 )\
                \
                
\f1\i \cf6 \strokec6 # Product categories
\f0\i0 \cf4 \strokec4 \
                product \cf8 \strokec8 =\cf4 \strokec4  np.random.choice([\cf2 \strokec2 'Electronics'\cf4 \strokec4 , \cf2 \strokec2 'Clothing'\cf4 \strokec4 , \cf2 \strokec2 'Home & Garden'\cf4 \strokec4 , \
                                          \cf2 \strokec2 'Sports'\cf4 \strokec4 , \cf2 \strokec2 'Books'\cf4 \strokec4 , \cf2 \strokec2 'Food & Beverage'\cf4 \strokec4 ])\
                \
                transactions.append(\{\
                    \cf2 \strokec2 'customer_id'\cf4 \strokec4 : cust_id,\
                    \cf2 \strokec2 'transaction_date'\cf4 \strokec4 : transaction_date,\
                    \cf2 \strokec2 'amount'\cf4 \strokec4 : \cf2 \strokec2 round\cf4 \strokec4 (amount, \cf7 \strokec7 2\cf4 \strokec4 ),\
                    \cf2 \strokec2 'product_category'\cf4 \strokec4 : product\
                \})\
        \
        self.customer_data \cf8 \strokec8 =\cf4 \strokec4  pd.DataFrame(transactions)\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Generated \cf4 \strokec4 \{\cf2 \strokec2 len\cf4 \strokec4 (transactions)\}\cf2 \strokec2  transactions for \cf4 \strokec4 \{n_customers\}\cf2 \strokec2  customers"\cf4 \strokec4 )\
        \cf5 \strokec5 return\cf4 \strokec4  self.customer_data\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf8 \strokec8 calculate_rfm_features\cf4 \strokec4 (self):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Calculate RFM (Recency, Frequency, Monetary) features\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nCalculating RFM features..."\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Set reference date as today
\f0\i0 \cf4 \strokec4 \
        reference_date \cf8 \strokec8 =\cf4 \strokec4  datetime.now()\
        \
        
\f1\i \cf6 \strokec6 # Calculate RFM metrics
\f0\i0 \cf4 \strokec4 \
        rfm \cf8 \strokec8 =\cf4 \strokec4  self.customer_data.groupby(\cf2 \strokec2 'customer_id'\cf4 \strokec4 ).agg(\{\
            \cf2 \strokec2 'transaction_date'\cf4 \strokec4 : \cf5 \strokec5 lambda\cf4 \strokec4  x: (reference_date \cf8 \strokec8 -\cf4 \strokec4  x.\cf2 \strokec2 max\cf4 \strokec4 ()).days,  
\f1\i \cf6 \strokec6 # Recency
\f0\i0 \cf4 \strokec4 \
            \cf2 \strokec2 'customer_id'\cf4 \strokec4 : \cf2 \strokec2 'count'\cf4 \strokec4 ,  
\f1\i \cf6 \strokec6 # Frequency
\f0\i0 \cf4 \strokec4 \
            \cf2 \strokec2 'amount'\cf4 \strokec4 : \cf2 \strokec2 'sum'\cf4 \strokec4   
\f1\i \cf6 \strokec6 # Monetary
\f0\i0 \cf4 \strokec4 \
        \})\
        \
        rfm.columns \cf8 \strokec8 =\cf4 \strokec4  [\cf2 \strokec2 'Recency'\cf4 \strokec4 , \cf2 \strokec2 'Frequency'\cf4 \strokec4 , \cf2 \strokec2 'Monetary'\cf4 \strokec4 ]\
        \
        
\f1\i \cf6 \strokec6 # Add additional features
\f0\i0 \cf4 \strokec4 \
        rfm[\cf2 \strokec2 'Average_Purchase_Value'\cf4 \strokec4 ] \cf8 \strokec8 =\cf4 \strokec4  self.customer_data.groupby(\cf2 \strokec2 'customer_id'\cf4 \strokec4 )[\cf2 \strokec2 'amount'\cf4 \strokec4 ].mean()\
        rfm[\cf2 \strokec2 'Purchase_Variance'\cf4 \strokec4 ] \cf8 \strokec8 =\cf4 \strokec4  self.customer_data.groupby(\cf2 \strokec2 'customer_id'\cf4 \strokec4 )[\cf2 \strokec2 'amount'\cf4 \strokec4 ].std().fillna(\cf7 \strokec7 0\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Calculate engagement score
\f0\i0 \cf4 \strokec4 \
        rfm[\cf2 \strokec2 'Engagement_Score'\cf4 \strokec4 ] \cf8 \strokec8 =\cf4 \strokec4  (\
            (\cf7 \strokec7 365\cf4 \strokec4  \cf8 \strokec8 -\cf4 \strokec4  rfm[\cf2 \strokec2 'Recency'\cf4 \strokec4 ]) \cf8 \strokec8 /\cf4 \strokec4  \cf7 \strokec7 365\cf4 \strokec4  \cf8 \strokec8 *\cf4 \strokec4  \cf7 \strokec7 0.3\cf4 \strokec4  \cf8 \strokec8 +\cf4 \strokec4 \
            np.log1p(rfm[\cf2 \strokec2 'Frequency'\cf4 \strokec4 ]) \cf8 \strokec8 /\cf4 \strokec4  np.log1p(rfm[\cf2 \strokec2 'Frequency'\cf4 \strokec4 ].\cf2 \strokec2 max\cf4 \strokec4 ()) \cf8 \strokec8 *\cf4 \strokec4  \cf7 \strokec7 0.4\cf4 \strokec4  \cf8 \strokec8 +\cf4 \strokec4 \
            np.log1p(rfm[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ]) \cf8 \strokec8 /\cf4 \strokec4  np.log1p(rfm[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ].\cf2 \strokec2 max\cf4 \strokec4 ()) \cf8 \strokec8 *\cf4 \strokec4  \cf7 \strokec7 0.3\cf4 \strokec4 \
        )\
        \
        self.rfm_data \cf8 \strokec8 =\cf4 \strokec4  rfm\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"RFM features calculated for \cf4 \strokec4 \{\cf2 \strokec2 len\cf4 \strokec4 (rfm)\}\cf2 \strokec2  customers"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nRFM Summary Statistics:"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (rfm.describe())\
        \
        \cf5 \strokec5 return\cf4 \strokec4  rfm\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf8 \strokec8 detect_anomalies\cf4 \strokec4 (self):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Apply anomaly detection to identify outliers in customer behavior\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nApplying anomaly detection..."\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Prepare features for anomaly detection
\f0\i0 \cf4 \strokec4 \
        features \cf8 \strokec8 =\cf4 \strokec4  self.rfm_data[[\cf2 \strokec2 'Recency'\cf4 \strokec4 , \cf2 \strokec2 'Frequency'\cf4 \strokec4 , \cf2 \strokec2 'Monetary'\cf4 \strokec4 ]].copy()\
        features_scaled \cf8 \strokec8 =\cf4 \strokec4  self.scaler.fit_transform(features)\
        \
        
\f1\i \cf6 \strokec6 # Detect anomalies
\f0\i0 \cf4 \strokec4 \
        anomaly_labels \cf8 \strokec8 =\cf4 \strokec4  self.anomaly_detector.fit_predict(features_scaled)\
        self.rfm_data[\cf2 \strokec2 'Is_Anomaly'\cf4 \strokec4 ] \cf8 \strokec8 =\cf4 \strokec4  anomaly_labels \cf8 \strokec8 ==\cf4 \strokec4  \cf8 \strokec8 -\cf7 \strokec7 1\cf4 \strokec4 \
        \
        n_anomalies \cf8 \strokec8 =\cf4 \strokec4  \cf2 \strokec2 sum\cf4 \strokec4 (self.rfm_data[\cf2 \strokec2 'Is_Anomaly'\cf4 \strokec4 ])\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Detected \cf4 \strokec4 \{n_anomalies\}\cf2 \strokec2  anomalous customers (\cf4 \strokec4 \{n_anomalies\cf8 \strokec8 /\cf2 \strokec2 len\cf4 \strokec4 (self.rfm_data)\cf8 \strokec8 *\cf7 \strokec7 100\cf4 \strokec4 :.1f\}\cf2 \strokec2 %)"\cf4 \strokec4 )\
        \
        \cf5 \strokec5 return\cf4 \strokec4  self.rfm_data\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf8 \strokec8 find_optimal_clusters\cf4 \strokec4 (self, max_k\cf8 \strokec8 =\cf7 \strokec7 10\cf4 \strokec4 ):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Find optimal number of clusters using Elbow Method and Silhouette Score\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nDetermining optimal number of clusters..."\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Filter out anomalies for clustering
\f0\i0 \cf4 \strokec4 \
        normal_customers \cf8 \strokec8 =\cf4 \strokec4  self.rfm_data[\cf8 \strokec8 ~\cf4 \strokec4 self.rfm_data[\cf2 \strokec2 'Is_Anomaly'\cf4 \strokec4 ]].copy()\
        features \cf8 \strokec8 =\cf4 \strokec4  normal_customers[[\cf2 \strokec2 'Recency'\cf4 \strokec4 , \cf2 \strokec2 'Frequency'\cf4 \strokec4 , \cf2 \strokec2 'Monetary'\cf4 \strokec4 ]].copy()\
        features_scaled \cf8 \strokec8 =\cf4 \strokec4  self.scaler.fit_transform(features)\
        \
        inertias \cf8 \strokec8 =\cf4 \strokec4  []\
        silhouette_scores \cf8 \strokec8 =\cf4 \strokec4  []\
        k_range \cf8 \strokec8 =\cf4 \strokec4  \cf2 \strokec2 range\cf4 \strokec4 (\cf7 \strokec7 2\cf4 \strokec4 , max_k \cf8 \strokec8 +\cf4 \strokec4  \cf7 \strokec7 1\cf4 \strokec4 )\
        \
        \cf5 \strokec5 for\cf4 \strokec4  k \cf5 \strokec5 in\cf4 \strokec4  k_range:\
            kmeans \cf8 \strokec8 =\cf4 \strokec4  KMeans(n_clusters\cf8 \strokec8 =\cf4 \strokec4 k, random_state\cf8 \strokec8 =\cf7 \strokec7 42\cf4 \strokec4 , n_init\cf8 \strokec8 =\cf7 \strokec7 10\cf4 \strokec4 )\
            labels \cf8 \strokec8 =\cf4 \strokec4  kmeans.fit_predict(features_scaled)\
            \
            inertias.append(kmeans.inertia_)\
            sil_score \cf8 \strokec8 =\cf4 \strokec4  silhouette_score(features_scaled, labels)\
            silhouette_scores.append(sil_score)\
            \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"K=\cf4 \strokec4 \{k\}\cf2 \strokec2 : Silhouette Score = \cf4 \strokec4 \{sil_score:.3f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Plot results
\f0\i0 \cf4 \strokec4 \
        fig, (ax1, ax2) \cf8 \strokec8 =\cf4 \strokec4  plt.subplots(\cf7 \strokec7 1\cf4 \strokec4 , \cf7 \strokec7 2\cf4 \strokec4 , figsize\cf8 \strokec8 =\cf4 \strokec4 (\cf7 \strokec7 14\cf4 \strokec4 , \cf7 \strokec7 5\cf4 \strokec4 ))\
        \
        ax1.plot(k_range, inertias, \cf2 \strokec2 'bo-'\cf4 \strokec4 )\
        ax1.set_xlabel(\cf2 \strokec2 'Number of Clusters (K)'\cf4 \strokec4 )\
        ax1.set_ylabel(\cf2 \strokec2 'Inertia'\cf4 \strokec4 )\
        ax1.set_title(\cf2 \strokec2 'Elbow Method for Optimal K'\cf4 \strokec4 )\
        ax1.grid(\cf7 \strokec7 True\cf4 \strokec4 )\
        \
        ax2.plot(k_range, silhouette_scores, \cf2 \strokec2 'ro-'\cf4 \strokec4 )\
        ax2.set_xlabel(\cf2 \strokec2 'Number of Clusters (K)'\cf4 \strokec4 )\
        ax2.set_ylabel(\cf2 \strokec2 'Silhouette Score'\cf4 \strokec4 )\
        ax2.set_title(\cf2 \strokec2 'Silhouette Score vs Number of Clusters'\cf4 \strokec4 )\
        ax2.grid(\cf7 \strokec7 True\cf4 \strokec4 )\
        \
        plt.tight_layout()\
        plt.savefig(\cf2 \strokec2 'optimal_clusters_analysis.png'\cf4 \strokec4 , dpi\cf8 \strokec8 =\cf7 \strokec7 300\cf4 \strokec4 , bbox_inches\cf8 \strokec8 =\cf2 \strokec2 'tight'\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nSaved: optimal_clusters_analysis.png"\cf4 \strokec4 )\
        \
        \cf5 \strokec5 return\cf4 \strokec4  silhouette_scores\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf8 \strokec8 perform_clustering\cf4 \strokec4 (self, n_clusters\cf8 \strokec8 =\cf7 \strokec7 4\cf4 \strokec4 ):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Perform K-Means clustering on RFM features\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\nPerforming K-Means clustering with \cf4 \strokec4 \{n_clusters\}\cf2 \strokec2  clusters..."\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Filter out anomalies
\f0\i0 \cf4 \strokec4 \
        normal_customers \cf8 \strokec8 =\cf4 \strokec4  self.rfm_data[\cf8 \strokec8 ~\cf4 \strokec4 self.rfm_data[\cf2 \strokec2 'Is_Anomaly'\cf4 \strokec4 ]].copy()\
        features \cf8 \strokec8 =\cf4 \strokec4  normal_customers[[\cf2 \strokec2 'Recency'\cf4 \strokec4 , \cf2 \strokec2 'Frequency'\cf4 \strokec4 , \cf2 \strokec2 'Monetary'\cf4 \strokec4 ]].copy()\
        \
        
\f1\i \cf6 \strokec6 # Scale features
\f0\i0 \cf4 \strokec4 \
        features_scaled \cf8 \strokec8 =\cf4 \strokec4  self.scaler.fit_transform(features)\
        \
        
\f1\i \cf6 \strokec6 # Perform clustering
\f0\i0 \cf4 \strokec4 \
        self.kmeans_model \cf8 \strokec8 =\cf4 \strokec4  KMeans(n_clusters\cf8 \strokec8 =\cf4 \strokec4 n_clusters, random_state\cf8 \strokec8 =\cf7 \strokec7 42\cf4 \strokec4 , n_init\cf8 \strokec8 =\cf7 \strokec7 10\cf4 \strokec4 )\
        cluster_labels \cf8 \strokec8 =\cf4 \strokec4  self.kmeans_model.fit_predict(features_scaled)\
        \
        
\f1\i \cf6 \strokec6 # Calculate metrics
\f0\i0 \cf4 \strokec4 \
        silhouette_avg \cf8 \strokec8 =\cf4 \strokec4  silhouette_score(features_scaled, cluster_labels)\
        db_score \cf8 \strokec8 =\cf4 \strokec4  davies_bouldin_score(features_scaled, cluster_labels)\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\nClustering Performance Metrics:"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Silhouette Score: \cf4 \strokec4 \{silhouette_avg:.3f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Davies-Bouldin Index: \cf4 \strokec4 \{db_score:.3f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Add cluster labels
\f0\i0 \cf4 \strokec4 \
        normal_customers[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ] \cf8 \strokec8 =\cf4 \strokec4  cluster_labels\
        \
        
\f1\i \cf6 \strokec6 # Handle anomalies separately
\f0\i0 \cf4 \strokec4 \
        anomaly_customers \cf8 \strokec8 =\cf4 \strokec4  self.rfm_data[self.rfm_data[\cf2 \strokec2 'Is_Anomaly'\cf4 \strokec4 ]].copy()\
        anomaly_customers[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ] \cf8 \strokec8 =\cf4 \strokec4  \cf8 \strokec8 -\cf7 \strokec7 1\cf4 \strokec4   
\f1\i \cf6 \strokec6 # Special label for anomalies
\f0\i0 \cf4 \strokec4 \
        \
        
\f1\i \cf6 \strokec6 # Combine back
\f0\i0 \cf4 \strokec4 \
        self.segmented_data \cf8 \strokec8 =\cf4 \strokec4  pd.concat([normal_customers, anomaly_customers])\
        \
        \cf5 \strokec5 return\cf4 \strokec4  silhouette_avg, cluster_labels\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf8 \strokec8 analyze_segments\cf4 \strokec4 (self):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Analyze and profile each customer segment\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nAnalyzing customer segments..."\cf4 \strokec4 )\
        \
        segment_profiles \cf8 \strokec8 =\cf4 \strokec4  []\
        \
        \cf5 \strokec5 for\cf4 \strokec4  cluster \cf5 \strokec5 in\cf4 \strokec4  \cf2 \strokec2 sorted\cf4 \strokec4 (self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ].unique()):\
            \cf5 \strokec5 if\cf4 \strokec4  cluster \cf8 \strokec8 ==\cf4 \strokec4  \cf8 \strokec8 -\cf7 \strokec7 1\cf4 \strokec4 :\
                segment_name \cf8 \strokec8 =\cf4 \strokec4  \cf2 \strokec2 "Anomalies"\cf4 \strokec4 \
                segment_data \cf8 \strokec8 =\cf4 \strokec4  self.segmented_data[self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ] \cf8 \strokec8 ==\cf4 \strokec4  \cf8 \strokec8 -\cf7 \strokec7 1\cf4 \strokec4 ]\
            \cf5 \strokec5 else\cf4 \strokec4 :\
                segment_data \cf8 \strokec8 =\cf4 \strokec4  self.segmented_data[self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ] \cf8 \strokec8 ==\cf4 \strokec4  cluster]\
                \
                
\f1\i \cf6 \strokec6 # Profile segment based on RFM characteristics
\f0\i0 \cf4 \strokec4 \
                avg_recency \cf8 \strokec8 =\cf4 \strokec4  segment_data[\cf2 \strokec2 'Recency'\cf4 \strokec4 ].mean()\
                avg_frequency \cf8 \strokec8 =\cf4 \strokec4  segment_data[\cf2 \strokec2 'Frequency'\cf4 \strokec4 ].mean()\
                avg_monetary \cf8 \strokec8 =\cf4 \strokec4  segment_data[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ].mean()\
                \
                \cf5 \strokec5 if\cf4 \strokec4  avg_recency \cf8 \strokec8 <\cf4 \strokec4  \cf7 \strokec7 90\cf4 \strokec4  \cf5 \strokec5 and\cf4 \strokec4  avg_monetary \cf8 \strokec8 >\cf4 \strokec4  segment_data[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ].median():\
                    segment_name \cf8 \strokec8 =\cf4 \strokec4  \cf2 \strokec2 f"Champions (Cluster \cf4 \strokec4 \{cluster\}\cf2 \strokec2 )"\cf4 \strokec4 \
                \cf5 \strokec5 elif\cf4 \strokec4  avg_frequency \cf8 \strokec8 >\cf4 \strokec4  segment_data[\cf2 \strokec2 'Frequency'\cf4 \strokec4 ].median():\
                    segment_name \cf8 \strokec8 =\cf4 \strokec4  \cf2 \strokec2 f"Loyal Customers (Cluster \cf4 \strokec4 \{cluster\}\cf2 \strokec2 )"\cf4 \strokec4 \
                \cf5 \strokec5 elif\cf4 \strokec4  avg_recency \cf8 \strokec8 <\cf4 \strokec4  \cf7 \strokec7 180\cf4 \strokec4 :\
                    segment_name \cf8 \strokec8 =\cf4 \strokec4  \cf2 \strokec2 f"Potential Loyalists (Cluster \cf4 \strokec4 \{cluster\}\cf2 \strokec2 )"\cf4 \strokec4 \
                \cf5 \strokec5 else\cf4 \strokec4 :\
                    segment_name \cf8 \strokec8 =\cf4 \strokec4  \cf2 \strokec2 f"At Risk (Cluster \cf4 \strokec4 \{cluster\}\cf2 \strokec2 )"\cf4 \strokec4 \
            \
            profile \cf8 \strokec8 =\cf4 \strokec4  \{\
                \cf2 \strokec2 'Segment'\cf4 \strokec4 : segment_name,\
                \cf2 \strokec2 'Cluster_ID'\cf4 \strokec4 : cluster,\
                \cf2 \strokec2 'Count'\cf4 \strokec4 : \cf2 \strokec2 len\cf4 \strokec4 (segment_data),\
                \cf2 \strokec2 'Percentage'\cf4 \strokec4 : \cf2 \strokec2 len\cf4 \strokec4 (segment_data) \cf8 \strokec8 /\cf4 \strokec4  \cf2 \strokec2 len\cf4 \strokec4 (self.segmented_data) \cf8 \strokec8 *\cf4 \strokec4  \cf7 \strokec7 100\cf4 \strokec4 ,\
                \cf2 \strokec2 'Avg_Recency'\cf4 \strokec4 : segment_data[\cf2 \strokec2 'Recency'\cf4 \strokec4 ].mean(),\
                \cf2 \strokec2 'Avg_Frequency'\cf4 \strokec4 : segment_data[\cf2 \strokec2 'Frequency'\cf4 \strokec4 ].mean(),\
                \cf2 \strokec2 'Avg_Monetary'\cf4 \strokec4 : segment_data[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ].mean(),\
                \cf2 \strokec2 'Total_Revenue'\cf4 \strokec4 : segment_data[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ].\cf2 \strokec2 sum\cf4 \strokec4 (),\
                \cf2 \strokec2 'Avg_Purchase_Value'\cf4 \strokec4 : segment_data[\cf2 \strokec2 'Average_Purchase_Value'\cf4 \strokec4 ].mean(),\
                \cf2 \strokec2 'Engagement_Score'\cf4 \strokec4 : segment_data[\cf2 \strokec2 'Engagement_Score'\cf4 \strokec4 ].mean()\
            \}\
            segment_profiles.append(profile)\
        \
        segment_df \cf8 \strokec8 =\cf4 \strokec4  pd.DataFrame(segment_profiles)\
        segment_df \cf8 \strokec8 =\cf4 \strokec4  segment_df.sort_values(\cf2 \strokec2 'Total_Revenue'\cf4 \strokec4 , ascending\cf8 \strokec8 =\cf7 \strokec7 False\cf4 \strokec4 )\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n"\cf4 \strokec4  \cf8 \strokec8 +\cf4 \strokec4  \cf2 \strokec2 "="\cf8 \strokec8 *\cf7 \strokec7 80\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "CUSTOMER SEGMENT PROFILES"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf8 \strokec8 *\cf7 \strokec7 80\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (segment_df.to_string(index\cf8 \strokec8 =\cf7 \strokec7 False\cf4 \strokec4 ))\
        \
        \cf5 \strokec5 return\cf4 \strokec4  segment_df\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf8 \strokec8 generate_marketing_insights\cf4 \strokec4 (self):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Generate actionable marketing insights for each segment\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n"\cf4 \strokec4  \cf8 \strokec8 +\cf4 \strokec4  \cf2 \strokec2 "="\cf8 \strokec8 *\cf7 \strokec7 80\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "ACTIONABLE MARKETING INSIGHTS"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf8 \strokec8 *\cf7 \strokec7 80\cf4 \strokec4 )\
        \
        insights \cf8 \strokec8 =\cf4 \strokec4  []\
        \
        \cf5 \strokec5 for\cf4 \strokec4  cluster \cf5 \strokec5 in\cf4 \strokec4  \cf2 \strokec2 sorted\cf4 \strokec4 (self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ].unique()):\
            \cf5 \strokec5 if\cf4 \strokec4  cluster \cf8 \strokec8 ==\cf4 \strokec4  \cf8 \strokec8 -\cf7 \strokec7 1\cf4 \strokec4 :\
                \cf5 \strokec5 continue\cf4 \strokec4 \
                \
            segment_data \cf8 \strokec8 =\cf4 \strokec4  self.segmented_data[self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ] \cf8 \strokec8 ==\cf4 \strokec4  cluster]\
            \
            avg_recency \cf8 \strokec8 =\cf4 \strokec4  segment_data[\cf2 \strokec2 'Recency'\cf4 \strokec4 ].mean()\
            avg_frequency \cf8 \strokec8 =\cf4 \strokec4  segment_data[\cf2 \strokec2 'Frequency'\cf4 \strokec4 ].mean()\
            avg_monetary \cf8 \strokec8 =\cf4 \strokec4  segment_data[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ].mean()\
            revenue_contribution \cf8 \strokec8 =\cf4 \strokec4  segment_data[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ].\cf2 \strokec2 sum\cf4 \strokec4 () \cf8 \strokec8 /\cf4 \strokec4  self.segmented_data[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ].\cf2 \strokec2 sum\cf4 \strokec4 () \cf8 \strokec8 *\cf4 \strokec4  \cf7 \strokec7 100\cf4 \strokec4 \
            \
            \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\\n--- CLUSTER \cf4 \strokec4 \{cluster\}\cf2 \strokec2  ---"\cf4 \strokec4 )\
            \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Size: \cf4 \strokec4 \{\cf2 \strokec2 len\cf4 \strokec4 (segment_data)\}\cf2 \strokec2  customers (\cf4 \strokec4 \{\cf2 \strokec2 len\cf4 \strokec4 (segment_data)\cf8 \strokec8 /\cf2 \strokec2 len\cf4 \strokec4 (self.segmented_data)\cf8 \strokec8 *\cf7 \strokec7 100\cf4 \strokec4 :.1f\}\cf2 \strokec2 %)"\cf4 \strokec4 )\
            \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Revenue Contribution: \cf4 \strokec4 \{revenue_contribution:.1f\}\cf2 \strokec2 %"\cf4 \strokec4 )\
            \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"Avg Recency: \cf4 \strokec4 \{avg_recency:.0f\}\cf2 \strokec2  days | Avg Frequency: \cf4 \strokec4 \{avg_frequency:.1f\}\cf2 \strokec2  | Avg Monetary: $\cf4 \strokec4 \{avg_monetary:.2f\}\cf2 \strokec2 "\cf4 \strokec4 )\
            \
            
\f1\i \cf6 \strokec6 # Generate recommendations
\f0\i0 \cf4 \strokec4 \
            \cf5 \strokec5 if\cf4 \strokec4  avg_recency \cf8 \strokec8 <\cf4 \strokec4  \cf7 \strokec7 90\cf4 \strokec4  \cf5 \strokec5 and\cf4 \strokec4  avg_monetary \cf8 \strokec8 >\cf4 \strokec4  self.segmented_data[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ].median():\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n\uc0\u55357 \u56522  STRATEGY: Premium Retention"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\'95 VIP rewards program and exclusive offers"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\'95 Personalized product recommendations"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\'95 Early access to new products/services"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\'95 Expected Campaign ROI: 300-400% (High-value retention)"\cf4 \strokec4 )\
                \
            \cf5 \strokec5 elif\cf4 \strokec4  avg_frequency \cf8 \strokec8 >\cf4 \strokec4  self.segmented_data[\cf2 \strokec2 'Frequency'\cf4 \strokec4 ].median():\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n\uc0\u55357 \u56522  STRATEGY: Loyalty Enhancement"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\'95 Loyalty points and tiered benefits"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\'95 Bundle offers and cross-selling opportunities"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\'95 Engagement campaigns to increase purchase frequency"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\'95 Expected Campaign ROI: 200-300% (Loyalty optimization)"\cf4 \strokec4 )\
                \
            \cf5 \strokec5 elif\cf4 \strokec4  avg_recency \cf8 \strokec8 <\cf4 \strokec4  \cf7 \strokec7 180\cf4 \strokec4 :\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n\uc0\u55357 \u56522  STRATEGY: Engagement & Upsell"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\'95 Targeted email campaigns with special offers"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\'95 Product trials and limited-time discounts"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\'95 Re-engagement through personalized content"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\'95 Expected Campaign ROI: 150-250% (Growth potential)"\cf4 \strokec4 )\
                \
            \cf5 \strokec5 else\cf4 \strokec4 :\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n\uc0\u55357 \u56522  STRATEGY: Win-back & Reactivation"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\'95 Win-back campaigns with significant discounts"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\'95 Survey to understand churn reasons"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\'95 Personalized reactivation offers"\cf4 \strokec4 )\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\'95 Expected Campaign ROI: 100-150% (Churn prevention)"\cf4 \strokec4 )\
            \
            
\f1\i \cf6 \strokec6 # Pricing recommendations
\f0\i0 \cf4 \strokec4 \
            \cf5 \strokec5 if\cf4 \strokec4  avg_monetary \cf8 \strokec8 >\cf4 \strokec4  self.segmented_data[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ].median():\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\uc0\u55357 \u56496  PRICING: Premium pricing acceptable, focus on value-added services"\cf4 \strokec4 )\
            \cf5 \strokec5 else\cf4 \strokec4 :\
                \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\uc0\u55357 \u56496  PRICING: Price-sensitive segment, emphasize discounts and value deals"\cf4 \strokec4 )\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n"\cf4 \strokec4  \cf8 \strokec8 +\cf4 \strokec4  \cf2 \strokec2 "="\cf8 \strokec8 *\cf7 \strokec7 80\cf4 \strokec4 )\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf8 \strokec8 visualize_segments\cf4 \strokec4 (self):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Create comprehensive visualizations for segments\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nGenerating visualization dashboards..."\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Create figure with subplots
\f0\i0 \cf4 \strokec4 \
        fig \cf8 \strokec8 =\cf4 \strokec4  plt.figure(figsize\cf8 \strokec8 =\cf4 \strokec4 (\cf7 \strokec7 18\cf4 \strokec4 , \cf7 \strokec7 12\cf4 \strokec4 ))\
        gs \cf8 \strokec8 =\cf4 \strokec4  fig.add_gridspec(\cf7 \strokec7 3\cf4 \strokec4 , \cf7 \strokec7 3\cf4 \strokec4 , hspace\cf8 \strokec8 =\cf7 \strokec7 0.3\cf4 \strokec4 , wspace\cf8 \strokec8 =\cf7 \strokec7 0.3\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # 1. Cluster distribution
\f0\i0 \cf4 \strokec4 \
        ax1 \cf8 \strokec8 =\cf4 \strokec4  fig.add_subplot(gs[\cf7 \strokec7 0\cf4 \strokec4 , \cf7 \strokec7 0\cf4 \strokec4 ])\
        cluster_counts \cf8 \strokec8 =\cf4 \strokec4  self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ].value_counts().sort_index()\
        ax1.bar(\cf2 \strokec2 range\cf4 \strokec4 (\cf2 \strokec2 len\cf4 \strokec4 (cluster_counts)), cluster_counts.values, color\cf8 \strokec8 =\cf2 \strokec2 'steelblue'\cf4 \strokec4 )\
        ax1.set_xlabel(\cf2 \strokec2 'Cluster ID'\cf4 \strokec4 )\
        ax1.set_ylabel(\cf2 \strokec2 'Number of Customers'\cf4 \strokec4 )\
        ax1.set_title(\cf2 \strokec2 'Customer Distribution Across Clusters'\cf4 \strokec4 )\
        ax1.grid(\cf7 \strokec7 True\cf4 \strokec4 , alpha\cf8 \strokec8 =\cf7 \strokec7 0.3\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # 2. Revenue by cluster
\f0\i0 \cf4 \strokec4 \
        ax2 \cf8 \strokec8 =\cf4 \strokec4  fig.add_subplot(gs[\cf7 \strokec7 0\cf4 \strokec4 , \cf7 \strokec7 1\cf4 \strokec4 ])\
        revenue_by_cluster \cf8 \strokec8 =\cf4 \strokec4  self.segmented_data.groupby(\cf2 \strokec2 'Cluster'\cf4 \strokec4 )[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ].\cf2 \strokec2 sum\cf4 \strokec4 ().sort_index()\
        ax2.bar(\cf2 \strokec2 range\cf4 \strokec4 (\cf2 \strokec2 len\cf4 \strokec4 (revenue_by_cluster)), revenue_by_cluster.values, color\cf8 \strokec8 =\cf2 \strokec2 'green'\cf4 \strokec4 , alpha\cf8 \strokec8 =\cf7 \strokec7 0.7\cf4 \strokec4 )\
        ax2.set_xlabel(\cf2 \strokec2 'Cluster ID'\cf4 \strokec4 )\
        ax2.set_ylabel(\cf2 \strokec2 'Total Revenue ($)'\cf4 \strokec4 )\
        ax2.set_title(\cf2 \strokec2 'Revenue Contribution by Cluster'\cf4 \strokec4 )\
        ax2.grid(\cf7 \strokec7 True\cf4 \strokec4 , alpha\cf8 \strokec8 =\cf7 \strokec7 0.3\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # 3. RFM 3D scatter (using first 3 dimensions)
\f0\i0 \cf4 \strokec4 \
        ax3 \cf8 \strokec8 =\cf4 \strokec4  fig.add_subplot(gs[\cf7 \strokec7 0\cf4 \strokec4 , \cf7 \strokec7 2\cf4 \strokec4 ], projection\cf8 \strokec8 =\cf2 \strokec2 '3d'\cf4 \strokec4 )\
        \cf5 \strokec5 for\cf4 \strokec4  cluster \cf5 \strokec5 in\cf4 \strokec4  \cf2 \strokec2 sorted\cf4 \strokec4 (self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ].unique()):\
            \cf5 \strokec5 if\cf4 \strokec4  cluster \cf8 \strokec8 !=\cf4 \strokec4  \cf8 \strokec8 -\cf7 \strokec7 1\cf4 \strokec4 :\
                cluster_data \cf8 \strokec8 =\cf4 \strokec4  self.segmented_data[self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ] \cf8 \strokec8 ==\cf4 \strokec4  cluster]\
                ax3.scatter(cluster_data[\cf2 \strokec2 'Recency'\cf4 \strokec4 ], cluster_data[\cf2 \strokec2 'Frequency'\cf4 \strokec4 ], \
                           cluster_data[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ], label\cf8 \strokec8 =\cf2 \strokec2 f'Cluster \cf4 \strokec4 \{cluster\}\cf2 \strokec2 '\cf4 \strokec4 , alpha\cf8 \strokec8 =\cf7 \strokec7 0.6\cf4 \strokec4 )\
        ax3.set_xlabel(\cf2 \strokec2 'Recency (days)'\cf4 \strokec4 )\
        ax3.set_ylabel(\cf2 \strokec2 'Frequency'\cf4 \strokec4 )\
        ax3.set_zlabel(\cf2 \strokec2 'Monetary ($)'\cf4 \strokec4 )\
        ax3.set_title(\cf2 \strokec2 '3D Customer Segmentation View'\cf4 \strokec4 )\
        ax3.legend()\
        \
        
\f1\i \cf6 \strokec6 # 4. Average RFM by cluster (heatmap)
\f0\i0 \cf4 \strokec4 \
        ax4 \cf8 \strokec8 =\cf4 \strokec4  fig.add_subplot(gs[\cf7 \strokec7 1\cf4 \strokec4 , :])\
        rfm_by_cluster \cf8 \strokec8 =\cf4 \strokec4  self.segmented_data.groupby(\cf2 \strokec2 'Cluster'\cf4 \strokec4 )[[\cf2 \strokec2 'Recency'\cf4 \strokec4 , \cf2 \strokec2 'Frequency'\cf4 \strokec4 , \cf2 \strokec2 'Monetary'\cf4 \strokec4 ]].mean()\
        rfm_by_cluster_normalized \cf8 \strokec8 =\cf4 \strokec4  (rfm_by_cluster \cf8 \strokec8 -\cf4 \strokec4  rfm_by_cluster.\cf2 \strokec2 min\cf4 \strokec4 ()) \cf8 \strokec8 /\cf4 \strokec4  (rfm_by_cluster.\cf2 \strokec2 max\cf4 \strokec4 () \cf8 \strokec8 -\cf4 \strokec4  rfm_by_cluster.\cf2 \strokec2 min\cf4 \strokec4 ())\
        sns.heatmap(rfm_by_cluster_normalized.T, annot\cf8 \strokec8 =\cf7 \strokec7 True\cf4 \strokec4 , fmt\cf8 \strokec8 =\cf2 \strokec2 '.2f'\cf4 \strokec4 , cmap\cf8 \strokec8 =\cf2 \strokec2 'RdYlGn_r'\cf4 \strokec4 , ax\cf8 \strokec8 =\cf4 \strokec4 ax4, cbar_kws\cf8 \strokec8 =\cf4 \strokec4 \{\cf2 \strokec2 'label'\cf4 \strokec4 : \cf2 \strokec2 'Normalized Value'\cf4 \strokec4 \})\
        ax4.set_title(\cf2 \strokec2 'Cluster RFM Profile Heatmap (Normalized)'\cf4 \strokec4 )\
        ax4.set_ylabel(\cf2 \strokec2 'RFM Metrics'\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # 5. Engagement score distribution
\f0\i0 \cf4 \strokec4 \
        ax5 \cf8 \strokec8 =\cf4 \strokec4  fig.add_subplot(gs[\cf7 \strokec7 2\cf4 \strokec4 , \cf7 \strokec7 0\cf4 \strokec4 ])\
        \cf5 \strokec5 for\cf4 \strokec4  cluster \cf5 \strokec5 in\cf4 \strokec4  \cf2 \strokec2 sorted\cf4 \strokec4 (self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ].unique()):\
            \cf5 \strokec5 if\cf4 \strokec4  cluster \cf8 \strokec8 !=\cf4 \strokec4  \cf8 \strokec8 -\cf7 \strokec7 1\cf4 \strokec4 :\
                cluster_data \cf8 \strokec8 =\cf4 \strokec4  self.segmented_data[self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ] \cf8 \strokec8 ==\cf4 \strokec4  cluster]\
                ax5.hist(cluster_data[\cf2 \strokec2 'Engagement_Score'\cf4 \strokec4 ], alpha\cf8 \strokec8 =\cf7 \strokec7 0.5\cf4 \strokec4 , label\cf8 \strokec8 =\cf2 \strokec2 f'Cluster \cf4 \strokec4 \{cluster\}\cf2 \strokec2 '\cf4 \strokec4 , bins\cf8 \strokec8 =\cf7 \strokec7 20\cf4 \strokec4 )\
        ax5.set_xlabel(\cf2 \strokec2 'Engagement Score'\cf4 \strokec4 )\
        ax5.set_ylabel(\cf2 \strokec2 'Frequency'\cf4 \strokec4 )\
        ax5.set_title(\cf2 \strokec2 'Engagement Score Distribution by Cluster'\cf4 \strokec4 )\
        ax5.legend()\
        ax5.grid(\cf7 \strokec7 True\cf4 \strokec4 , alpha\cf8 \strokec8 =\cf7 \strokec7 0.3\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # 6. Frequency vs Monetary
\f0\i0 \cf4 \strokec4 \
        ax6 \cf8 \strokec8 =\cf4 \strokec4  fig.add_subplot(gs[\cf7 \strokec7 2\cf4 \strokec4 , \cf7 \strokec7 1\cf4 \strokec4 ])\
        \cf5 \strokec5 for\cf4 \strokec4  cluster \cf5 \strokec5 in\cf4 \strokec4  \cf2 \strokec2 sorted\cf4 \strokec4 (self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ].unique()):\
            \cf5 \strokec5 if\cf4 \strokec4  cluster \cf8 \strokec8 !=\cf4 \strokec4  \cf8 \strokec8 -\cf7 \strokec7 1\cf4 \strokec4 :\
                cluster_data \cf8 \strokec8 =\cf4 \strokec4  self.segmented_data[self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ] \cf8 \strokec8 ==\cf4 \strokec4  cluster]\
                ax6.scatter(cluster_data[\cf2 \strokec2 'Frequency'\cf4 \strokec4 ], cluster_data[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ], \
                           label\cf8 \strokec8 =\cf2 \strokec2 f'Cluster \cf4 \strokec4 \{cluster\}\cf2 \strokec2 '\cf4 \strokec4 , alpha\cf8 \strokec8 =\cf7 \strokec7 0.6\cf4 \strokec4 )\
        ax6.set_xlabel(\cf2 \strokec2 'Frequency (# Transactions)'\cf4 \strokec4 )\
        ax6.set_ylabel(\cf2 \strokec2 'Monetary Value ($)'\cf4 \strokec4 )\
        ax6.set_title(\cf2 \strokec2 'Frequency vs Monetary Value by Cluster'\cf4 \strokec4 )\
        ax6.legend()\
        ax6.grid(\cf7 \strokec7 True\cf4 \strokec4 , alpha\cf8 \strokec8 =\cf7 \strokec7 0.3\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # 7. Recency vs Monetary
\f0\i0 \cf4 \strokec4 \
        ax7 \cf8 \strokec8 =\cf4 \strokec4  fig.add_subplot(gs[\cf7 \strokec7 2\cf4 \strokec4 , \cf7 \strokec7 2\cf4 \strokec4 ])\
        \cf5 \strokec5 for\cf4 \strokec4  cluster \cf5 \strokec5 in\cf4 \strokec4  \cf2 \strokec2 sorted\cf4 \strokec4 (self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ].unique()):\
            \cf5 \strokec5 if\cf4 \strokec4  cluster \cf8 \strokec8 !=\cf4 \strokec4  \cf8 \strokec8 -\cf7 \strokec7 1\cf4 \strokec4 :\
                cluster_data \cf8 \strokec8 =\cf4 \strokec4  self.segmented_data[self.segmented_data[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ] \cf8 \strokec8 ==\cf4 \strokec4  cluster]\
                ax7.scatter(cluster_data[\cf2 \strokec2 'Recency'\cf4 \strokec4 ], cluster_data[\cf2 \strokec2 'Monetary'\cf4 \strokec4 ], \
                           label\cf8 \strokec8 =\cf2 \strokec2 f'Cluster \cf4 \strokec4 \{cluster\}\cf2 \strokec2 '\cf4 \strokec4 , alpha\cf8 \strokec8 =\cf7 \strokec7 0.6\cf4 \strokec4 )\
        ax7.set_xlabel(\cf2 \strokec2 'Recency (days)'\cf4 \strokec4 )\
        ax7.set_ylabel(\cf2 \strokec2 'Monetary Value ($)'\cf4 \strokec4 )\
        ax7.set_title(\cf2 \strokec2 'Recency vs Monetary Value by Cluster'\cf4 \strokec4 )\
        ax7.legend()\
        ax7.grid(\cf7 \strokec7 True\cf4 \strokec4 , alpha\cf8 \strokec8 =\cf7 \strokec7 0.3\cf4 \strokec4 )\
        \
        plt.savefig(\cf2 \strokec2 'customer_segmentation_dashboard.png'\cf4 \strokec4 , dpi\cf8 \strokec8 =\cf7 \strokec7 300\cf4 \strokec4 , bbox_inches\cf8 \strokec8 =\cf2 \strokec2 'tight'\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "Saved: customer_segmentation_dashboard.png"\cf4 \strokec4 )\
        plt.close()\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf8 \strokec8 export_results\cf4 \strokec4 (self):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Export all results to CSV files for Tableau/Power BI integration\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\nExporting results for Tableau and Power BI..."\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Export main segmentation results
\f0\i0 \cf4 \strokec4 \
        self.segmented_data.to_csv(\cf2 \strokec2 'customer_segments.csv'\cf4 \strokec4 , index\cf8 \strokec8 =\cf7 \strokec7 True\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "Saved: customer_segments.csv"\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Export segment profiles
\f0\i0 \cf4 \strokec4 \
        segment_summary \cf8 \strokec8 =\cf4 \strokec4  self.analyze_segments()\
        segment_summary.to_csv(\cf2 \strokec2 'segment_profiles.csv'\cf4 \strokec4 , index\cf8 \strokec8 =\cf7 \strokec7 False\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "Saved: segment_profiles.csv"\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Export transaction-level data with segments
\f0\i0 \cf4 \strokec4 \
        transaction_with_segments \cf8 \strokec8 =\cf4 \strokec4  self.customer_data.merge(\
            self.segmented_data[[\cf2 \strokec2 'Cluster'\cf4 \strokec4 ]], \
            left_on\cf8 \strokec8 =\cf2 \strokec2 'customer_id'\cf4 \strokec4 , \
            right_index\cf8 \strokec8 =\cf7 \strokec7 True\cf4 \strokec4 \
        )\
        transaction_with_segments.to_csv(\cf2 \strokec2 'transactions_with_segments.csv'\cf4 \strokec4 , index\cf8 \strokec8 =\cf7 \strokec7 False\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "Saved: transactions_with_segments.csv"\cf4 \strokec4 )\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n\uc0\u9989  All files ready for Tableau and Power BI dashboard creation!"\cf4 \strokec4 )\
    \
    \cf5 \strokec5 def\cf4 \strokec4  \cf8 \strokec8 run_full_analysis\cf4 \strokec4 (self, n_customers\cf8 \strokec8 =\cf7 \strokec7 200\cf4 \strokec4 , n_clusters\cf8 \strokec8 =\cf7 \strokec7 4\cf4 \strokec4 ):\
        \cf2 \strokec2 """\
\pard\pardeftab720\partightenfactor0
\cf4 \strokec4         Execute the complete segmentation pipeline\
\pard\pardeftab720\partightenfactor0
\cf2 \strokec2         """\cf4 \strokec4 \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf8 \strokec8 *\cf7 \strokec7 80\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "CUSTOMERSENSE - CUSTOMER SEGMENTATION ENGINE"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "JPMorgan Challenge Project"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf8 \strokec8 *\cf7 \strokec7 80\cf4 \strokec4 )\
        \
        
\f1\i \cf6 \strokec6 # Step 1: Generate data
\f0\i0 \cf4 \strokec4 \
        self.generate_synthetic_data(n_customers)\
        \
        
\f1\i \cf6 \strokec6 # Step 2: Calculate RFM features
\f0\i0 \cf4 \strokec4 \
        self.calculate_rfm_features()\
        \
        
\f1\i \cf6 \strokec6 # Step 3: Detect anomalies
\f0\i0 \cf4 \strokec4 \
        self.detect_anomalies()\
        \
        
\f1\i \cf6 \strokec6 # Step 4: Find optimal clusters
\f0\i0 \cf4 \strokec4 \
        silhouette_scores \cf8 \strokec8 =\cf4 \strokec4  self.find_optimal_clusters()\
        \
        
\f1\i \cf6 \strokec6 # Step 5: Perform clustering
\f0\i0 \cf4 \strokec4 \
        silhouette_score, labels \cf8 \strokec8 =\cf4 \strokec4  self.perform_clustering(n_clusters)\
        \
        
\f1\i \cf6 \strokec6 # Step 6: Analyze segments
\f0\i0 \cf4 \strokec4 \
        segment_profiles \cf8 \strokec8 =\cf4 \strokec4  self.analyze_segments()\
        \
        
\f1\i \cf6 \strokec6 # Step 7: Generate insights
\f0\i0 \cf4 \strokec4 \
        self.generate_marketing_insights()\
        \
        
\f1\i \cf6 \strokec6 # Step 8: Create visualizations
\f0\i0 \cf4 \strokec4 \
        self.visualize_segments()\
        \
        
\f1\i \cf6 \strokec6 # Step 9: Export results
\f0\i0 \cf4 \strokec4 \
        self.export_results()\
        \
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "\\n"\cf4 \strokec4  \cf8 \strokec8 +\cf4 \strokec4  \cf2 \strokec2 "="\cf8 \strokec8 *\cf7 \strokec7 80\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "ANALYSIS COMPLETE!"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf8 \strokec8 *\cf7 \strokec7 80\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\uc0\u9989  Segmented \cf4 \strokec4 \{n_customers\}\cf2 \strokec2 + customers using RFM features and K-Means"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\uc0\u9989  Achieved silhouette score of \cf4 \strokec4 \{silhouette_score:.2f\}\cf2 \strokec2 "\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\uc0\u9989  Generated actionable insights for targeted marketing"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\uc0\u9989  Applied anomaly detection and statistical validation"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\uc0\u9989  Created visualizations for Tableau and Power BI dashboards"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 f"\uc0\u9989  Strengthened decision-making on pricing, retention, and campaign ROI"\cf4 \strokec4 )\
        \cf5 \strokec5 print\cf4 \strokec4 (\cf2 \strokec2 "="\cf8 \strokec8 *\cf7 \strokec7 80\cf4 \strokec4 )\
\
\
\pard\pardeftab720\partightenfactor0
\cf5 \strokec5 if\cf4 \strokec4  __name__ \cf8 \strokec8 ==\cf4 \strokec4  \cf2 \strokec2 "__main__"\cf4 \strokec4 :\
    
\f1\i \cf6 \strokec6 # Initialize and run the segmentation engine
\f0\i0 \cf4 \strokec4 \
    engine \cf8 \strokec8 =\cf4 \strokec4  CustomerSegmentationEngine()\
    engine.run_full_analysis(n_customers\cf8 \strokec8 =\cf7 \strokec7 200\cf4 \strokec4 , n_clusters\cf8 \strokec8 =\cf7 \strokec7 4\cf4 \strokec4 )}