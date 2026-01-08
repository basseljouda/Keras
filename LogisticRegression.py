
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

file_path="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
testSize = 0.2

data = pd.read_csv(file_path)

scale = StandardScaler()
scaled_columns = data.select_dtypes(include=['float64']).columns.to_list()
scaled_data = scale.fit_transform(data[scaled_columns])
#scaled_df = pd.DataFrame(scaled_data,columns=scaled_columns)
scaled_df = pd.DataFrame(scaled_data,columns=scale.get_feature_names_out(scaled_columns))
scaled_data = pd.concat([data.drop(columns=scaled_columns),scaled_df],axis=1)

encoder = OneHotEncoder(sparse_output=False,drop='first')
encoded_columns = scaled_data.select_dtypes(include='object').columns.to_list()
encoded_columns.remove('NObeyesdad')
encoder_data = encoder.fit_transform(scaled_data[encoded_columns])
encoder_df = pd.DataFrame(encoder_data,columns=encoder.get_feature_names_out(encoded_columns))
data_processed = pd.concat([scaled_data.drop(columns=encoded_columns),encoder_df],axis=1)

X = data_processed.drop('NObeyesdad',axis=1)
y = data_processed['NObeyesdad']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=testSize,random_state=42,stratify=y)
model_ova = LogisticRegression(multi_class='ovr',max_iter=1000)
model_ova.fit(X_train,y_train)
y_pred = model_ova.predict(X_test)

print(accuracy_score(y_test,y_pred))

feature = np.mean(np.abs(model_ova.coef_),axis=0)
plt.barh(X.columns,feature)
plt.title("Chart")
plt.xlabel("Importance")
plt.show()