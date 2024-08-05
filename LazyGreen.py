import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from lazypredict.Supervised import LazyRegressor, LazyClassifier

# Function to convert 'N/P' values to binary
def convert_np_to_binary(value):
    if value == 'P':
        return 1
    elif value == 'N':
        return 0
    else:
        return pd.NA

# Parsing command line arguments
parser = argparse.ArgumentParser(description='Run machine learning models on the specified target label.')
parser.add_argument('--file_path', type=str, required=True, help='The path to your CSV file')
parser.add_argument('--target_label', type=str, required=True, help='Target label for the prediction model')
parser.add_argument('--algorithm', type=str, choices=['LazyRegressor', 'LazyClassifier'], required=True, help='Algorithm type to use (LazyRegressor or LazyClassifier)')
args = parser.parse_args()

# Load dataset
data = pd.read_csv(args.file_path)

# Apply the binary conversion function
for column in ['HF183_pa', 'Rum2Bac_pa', 'DG3_pa', 'GFD_pa']:
    data[column] = data[column].apply(convert_np_to_binary)

# Impute numerical missing values
numerical_columns = data.select_dtypes(include=['number']).columns
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = numerical_imputer.fit_transform(data[numerical_columns])

# One-hot encode categorical data
categorical_columns = data.select_dtypes(include=['object']).columns
onehot_encoder = OneHotEncoder()
transformer = ColumnTransformer(transformers=[('cat', onehot_encoder, categorical_columns)], remainder='passthrough')
data_transformed = transformer.fit_transform(data)
transformed_columns = transformer.get_feature_names_out()

# Create DataFrame with encoded data
data_encoded = pd.DataFrame(data_transformed, columns=transformed_columns)

# Combine the four binary target labels into one
data_encoded['combined_target'] = data[['HF183_pa', 'Rum2Bac_pa', 'DG3_pa', 'GFD_pa']].max(axis=1)

# List target labels
target_labels = [col for col in data_encoded.columns if 'remainder__' in col]
target_labels.append('combined_target')
print("Available target labels:")
for label in target_labels:
    print(label)

# Exclude specific feature columns for prediction
X = data_encoded.drop(columns=target_labels)
y = data_encoded[args.target_label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using LazyPredict
if args.algorithm == 'LazyRegressor':
    clf = LazyRegressor(verbose=0, ignore_warnings=True, predictions=True)
else:
    clf = LazyClassifier(verbose=0, ignore_warnings=True, predictions=True)

models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Display results
print("Model Performance:")
print(models)
print("\nPredictions:")
print(predictions)
