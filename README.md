# 🧠 Customer Clustering Project — From Clusters to Campaigns

This project aims to segment credit card customers based on their financial behavior using unsupervised machine learning, specifically KMeans clustering. By analyzing patterns in spending, cash advances, payments, and purchase types, we were able to generate customer clusters that can be used for business insights, such as targeted marketing, rewards strategy, and credit risk profiling.

We began with a thorough exploratory data analysis (EDA) where we examined relationships between key variables. For instance, we looked into how a customer's balance relates to their balance frequency, how their payment habits align with their minimum due, and how installment purchases differ from one-off purchases. These comparisons helped us engineer features that better reflect behavioral tendencies and prepare the data for clustering.

After preprocessing and normalization, we applied Principal Component Analysis (PCA) to reduce dimensionality while preserving variance. The KMeans algorithm was then used to identify clusters, with the optimal number determined through the elbow method and silhouette score evaluation. We also analyzed the characteristics of each cluster to understand the behavioral profiles of our customer groups.

To support interactive data exploration, we built a **Streamlit app** for this project. Although still in development, the Streamlit dashboard allows future users to explore customer clusters, visualize patterns, and potentially apply filters for different variables. This is intended to serve as a prototype for integrating unsupervised learning into business-facing analytics tools.

You can access the deployed dashboard using the following link:

🔗 **Streamlit App:**  
(https://deltapurwadhikaappfinpro.streamlit.app/)

Additionally, we published a separate visualization of the project in Tableau Public. This interactive dashboard focuses on the customer segments, their average purchase behavior, and overall spending insights. It provides a storytelling layer for business audiences who may prefer point-and-click exploration over code.

🔗 **Tableau Public Dashboard:**  
(https://public.tableau.com/views/PWDFinalProjectCreditCardUserSegmentations/CreditCardSegmentations?:language=en-GB&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

! [Tableau Screenshot](https://github.com/PurwadhikaDev/DeltaGroup_DTI_OL_02_FinalProject/blob/main/Credit%20Card%20Segmentations.png)

This project demonstrates how clustering can uncover hidden behavioral patterns in customer data. Each identified group brings unique characteristics that businesses can act upon, such as promoting installment-friendly plans for one group or offering cash advance alternatives for another. In future iterations, this analysis could be extended with geographic or demographic data and improved by testing clustering methods like DBSCAN or Agglomerative Clustering.
