from copy import deepcopy
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.express as px
from sklearn.linear_model import Lasso, Ridge
from sklearn.linear_model import LinearRegression
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# 'Side' column:
# Fill blank samples with the most common value
# common_side = X['Side'].value_counts().idxmax()
# X['Side'].fillna(common_side, inplace=True)


# # 'Basic stage' processing before dummifying
# X.loc[X['Basic stage'].str.contains('Null', case=False),
# 'Basic stage'] = 'c - Clinical'
# X.loc[X['Basic stage'].str.contains('drr', case=False),
# 'Basic stage'] = 'r - Reccurent'
#

# Dummify above columns
# X_ = pd.get_dummies(X, columns=['Basic stage', 'Histological diagnosis',
# 								'Side'])

column_pattern = '[^a-zA-Z0-9\s-]+'
negative_patterns = '-|neg|0|1|1\+?|שלילי|meg|nec|akhkh|nef|nfg|nrg|no|heg|nag|=|o'
positive_patterns = 'pos|3\+?|חיובי|po|jhuch'
fish_patterns = '2|FISH|indeterm|בינוני'

columns_to_drop = ["Hospital", "User Name", "id-hushedinternalpatientid",
                   'Surgery date2', 'Surgery date1',
                   'Surgery date3', 'Surgery name1',
                   'Surgery name2',
                   'Surgery name3', 'Tumor depth', 'Tumor width',
                   'surgery before or after-Actual activity',
                   'surgery before or after-Activity date', "Form Name",
                   "Ivi -Lymphovascular invasion",
                   "Lymphatic penetration", 'Positive nodes', 'er',
                   'pr', 'Histological diagnosis', 'Basic stage', 'Side', "Diagnosis date"]


# Function to calculate the mean of percentage range
def calculate_mean_percentage(percentage):
    # Extract numbers from the range and convert to int
    numbers = re.findall(r'\d+', str(percentage))
    numbers = [int(num) for num in numbers]
    # Calculate the mean of the range
    return sum(numbers) / len(numbers) if len(numbers) else 0


def process_her2(X: pd.DataFrame):
    X['Her2'] = X['Her2'].fillna(
        1).astype(str)
    X['Her2'] = X['Her2'].fillna('1').astype(str)
    X['Her2'] = np.where(
        X['Her2'].str.contains(fish_patterns, case=False, na=False), '2',
        X['Her2'])
    X['Her2'] = np.where(
        X['Her2'].str.contains(negative_patterns, case=False, na=False), '1',
        X['Her2'])
    X['Her2'] = np.where(
        X['Her2'].str.contains(positive_patterns, case=False, na=False), '3',
        X['Her2'])
    X['Her2'] = np.where(~X['Her2'].isin(['1', '2', '3']), '1', X['Her2'])
    return X


def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    # Fix column names by removing Hebrew and irrelevant symbols
    column_dict = {col: re.sub(column_pattern, '', col).strip('- ') for col in
                   X.columns}
    X.rename(columns=column_dict, inplace=True)

    # 'Age' column:
    # Calculate the mean of all relevant samples, and replace outliersw ith it
    age_mean = X[(X['Age'] > 10) & (X['Age'] <= 110)]['Age'].mean()
    X.loc[(X['Age'] <= 10) | (X['Age'] > 110), 'Age'] = age_mean

    # 'Diagnosis date' column:
    # Change the Diagnosis date column to diagnosis year
    # X['Diagnosis date'] = pd.to_datetime(X['Diagnosis date'],
    #                                      format='%d/%m/%Y %H:%M')
    # X['Diagnosis Year'] = X['Diagnosis date'].dt.year.astype(int)

    # 'Stage' and 'T -Tumor mark (TNM)' columns:
    # Extract numbers and fill blanks with 0
    X['Stage'] = X['Stage'].str.extract(r'(\d+)').fillna('0').astype(int)
    X['T -Tumor mark TNM'] = X['T -Tumor mark TNM'].str.extract(
        r'(\d+)').fillna('0').astype(int)

    # 'Surgery sum', 'Nodes exam' columns:
    # Fill blanks with 0
    X['Surgery sum'] = X['Surgery sum'].fillna(0).astype(int)
    X['Nodes exam'] = X['Nodes exam'].fillna(0).astype(int)

    # 'N -lymph nodes mark TNM' column:
    # - Firstly, fill blanks with string '1'
    # - Then, replace degree score with corresponding number (i.e N1 -> 1)
    X['N -lymph nodes mark TNM'] = X['N -lymph nodes mark TNM'].fillna(
        1).astype(str)
    X.loc[X['N -lymph nodes mark TNM'].str.contains('NX|NAME|0|Not',
                                                    case=False),
          'N -lymph nodes mark TNM'] = 0
    X.loc[X['N -lymph nodes mark TNM'].str.contains('N1|ITC', na=False,
                                                    case=False),
          'N -lymph nodes mark TNM'] = 1
    X.loc[X['N -lymph nodes mark TNM'].str.contains('N2', na=False,
                                                    case=False),
          'N -lymph nodes mark TNM'] = 2
    X.loc[X['N -lymph nodes mark TNM'].str.contains('N3', na=False,
                                                    case=False),
          'N -lymph nodes mark TNM'] = 3
    X.loc[X['N -lymph nodes mark TNM'].str.contains('N4', na=False,
                                                    case=False),
          'N -lymph nodes mark TNM'] = 4

    # 'M -metastases mark TNM' column:
    # Same as above
    X['M -metastases mark TNM'] = X['M -metastases mark TNM'].fillna(
        0).astype(str)
    X.loc[X['M -metastases mark TNM'].str.contains('1', na=False),
          'M -metastases mark TNM'] = 1
    X.loc[X['M -metastases mark TNM'].str.contains('0|x|not', na=False,
                                                   case=False),
          'M -metastases mark TNM'] = 0

    # 'K167 protein' processing before dummifying
    X['KI67 protein'] = X['KI67 protein'].apply(calculate_mean_percentage)

    # 'Margin Type' column:
    # Replace Hebrew values with 0 or 1 (i.e ׳ללא׳ - > 0)
    X['Margin Type'] = X['Margin Type'].astype(str)  # Convert column to string type

    X.loc[X['Margin Type'].str.contains('ללא|נקיים',
                                        na=False), 'Margin Type'] = 0
    X.loc[X['Margin Type'].str.contains('נגועים',
                                        na=False), 'Margin Type'] = 1

    # 'Histopatological degree' column:
    # Replace degree score with corresponding number (i.e G1 -> 1);
    # processing before dummifying
    X['Histopatological degree'] = X['Histopatological degree'].astype(str)  # Convert column to string type

    X.loc[X['Histopatological degree'].str.contains('GX|Null', case=False),
          'Histopatological degree'] = 0
    X.loc[X['Histopatological degree'].str.contains('G1', na=False,
                                                    case=False),
          'Histopatological degree'] = 1
    X.loc[X['Histopatological degree'].str.contains('G2', na=False,
                                                    case=False),
          'Histopatological degree'] = 2
    X.loc[X['Histopatological degree'].str.contains('G3', na=False,
                                                    case=False),
          'Histopatological degree'] = 3
    X.loc[X['Histopatological degree'].str.contains('G4', na=False,
                                                    case=False),
          'Histopatological degree'] = 4

    # 'Her2' column:
    # Regex to remove outliers and classify
    process_her2(X)

    X.drop(columns=columns_to_drop, inplace=True)

    # if y was included - concatenate it, remove duplicates and return X, y
    if y is not None:
        X_ = pd.concat([X, y], axis=1)
        X_.drop_duplicates()
        return X_.iloc[:, :-1], X_.iloc[:, -1]

    return X


def perform_pca(data):
    # Create an instance of PCA and fit it to the standardized data
    kmeans = KMeans(n_clusters=4)  # Specify the number of clusters
    kmeans.fit(data)

    # Obtain the cluster labels for each data point
    cluster_labels = kmeans.labels_
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)

    # Create a DataFrame with the principal components
    principal_df = pd.DataFrame(data=principal_components,
                                columns=['PC1', 'PC2'])

    return principal_df, cluster_labels


def plot_2d_data(data, cluster_labels):
    data = np.array(data)
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA with Clustering')

    # Add color bar legend
    cbar = plt.colorbar()
    cbar.set_label('Cluster')
    plt.show()


def plot_eigenvalues(data, max_components=None):
    # Perform PCA
    pca = PCA(n_components=max_components)
    pca.fit(data)
    print("ffbfb")
    # Get the eigenvalues
    eigenvalues = pca.explained_variance_

    # Plot the eigenvalues
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Eigenvalues')
    plt.title('Eigenvalues as a function of Number of Principal Components')
    plt.grid(True)

    # Show the plot
    plt.show()


def perform_clustering_and_plot(data, n_clusters=3):
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_labels = kmeans.fit_predict(data)

    # Create a DataFrame with original data and cluster labels
    df = pd.DataFrame(data)
    df['Cluster'] = cluster_labels

    # Generate insights about the features
    cluster_stats = df.groupby('Cluster').describe()

    # Plot the clusters
    num_features = data.shape[1]
    fig, axes = plt.subplots(nrows=num_features, figsize=(8, 6 * num_features))
    fig.suptitle('Clustering Results')
    numeric_columns = df.select_dtypes(include=np.number).columns  # Select only numeric columns
    for i, feature in enumerate(numeric_columns[:-1]):  # Exclude the last column ('Cluster')
        ax = axes[i]
        ax.set_title(f'Feature {feature}')
        for cluster_label, cluster_data in df.groupby('Cluster'):
            ax.hist(cluster_data[feature], alpha=0.5, label=f'Cluster {cluster_label}')
        ax.legend()

    # Adjust the spacing between subplots
    fig.tight_layout()

    # Show the plot
    plt.show()

    # Return cluster statistics
    return cluster_stats


def find_correlation(data):
    # Calculate the correlation matrix
    correlation_matrix = data.corr()
    # Create a heatmap of the correlation matrix using Plotly
    fig = px.imshow(correlation_matrix)
    fig.update_layout(title="Correlation Matrix")
    fig.write_image("correlation_matrix.png")


def k_fold_regressor(X_train, y_train, X_test, y_test, cv, model):
    ids = np.arange(X_train.shape[0])
    # Randomly split samples into `cv` folds
    folds = np.array_split(ids, cv)
    test_score = .0
    for fold_ids in folds:
        train_msk = np.ones(len(X_train), dtype=bool)
        train_msk[fold_ids] = False
        fit = deepcopy(model).fit(X_train[train_msk], y_train[train_msk])
        test_score += mean_squared_error(y_test[fold_ids], fit.predict(X_test[fold_ids]))

    return test_score / cv


def erm_regressor(X_train, y_train, X_test, y_test, num, cv):
    ridge_range = np.linspace(0.0001, 2, num=num)
    lasso_range = np.linspace(0.0001, 2, num=num)
    ridge_test_errors = np.zeros(num)
    lasso_test_errors = np.zeros(num)
    for i, lam in enumerate(ridge_range):
        train_err = k_fold_regressor(X_train, y_train, X_test, y_test, cv, Ridge(alpha=lam))
        ridge_test_errors[i] = train_err
    for i, lam in enumerate(lasso_range):
        train_err = k_fold_regressor(X_train, y_train, X_test, y_test, cv, Lasso(alpha=lam))
        lasso_test_errors[i] = train_err
    best_ridge = ridge_range[np.argmin(ridge_test_errors)]
    best_lasso = lasso_range[np.argmin(lasso_test_errors)]
    return best_ridge, best_lasso


def check_simple_regressors(X_train, y_train, X_test, y_test, ridge_lam, lasso_lam, cv):
    min_mse = []
    models = []
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    reg_pred = reg_model.predict(X_test)
    reg_loss = mean_squared_error(y_test, reg_pred)
    min_mse.append(reg_loss)
    models.append("linear reg")
    # print(reg_loss)

    k_lin_reg = k_fold_regressor(X_train, y_train, X_test, y_test, cv, LinearRegression())
    min_mse.append(k_lin_reg)
    models.append("k-folds linear reg")

    ridge = Ridge(alpha=ridge_lam)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    ridge_loss = mean_squared_error(y_test, ridge_pred)
    min_mse.append(ridge_loss)
    models.append("ridge reg")

    ridge_k = k_fold_regressor(X_train, y_train, X_test, y_test, cv, Ridge(alpha=ridge_lam))
    min_mse.append(ridge_k)
    models.append("k-folds ridge")

    # print("ridge loss:", ridge_loss)

    lasso = Lasso(alpha=lasso_lam)
    lasso.fit(X_train, y_train)
    lasso_pred = lasso.predict(X_test)
    lasso_loss = mean_squared_error(y_test, lasso_pred)
    min_mse.append(lasso_loss)
    models.append("lasso reg")

    lasso_k = k_fold_regressor(X_train, y_train, X_test, y_test, cv, Lasso(alpha=lasso_lam))
    min_mse.append(lasso_k)
    models.append("k-folds lasso")
    # print("lasso loss:", lasso_loss)

    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train, y_train)
    rf_pred = rf_reg.predict(X_test)
    rf_loss = mean_squared_error(y_test, rf_pred)
    min_mse.append(rf_loss)
    models.append("random forest reg")

    # Calculate the average of the true y values
    average_y = np.mean(y_train)
    mean_loss = mean_squared_error(y_test, np.full_like(y_test, average_y))
    min_mse.append(mean_loss)
    models.append("average label")

    errors_df = pd.DataFrame(list(zip(models, min_mse)), columns=("models", "MSE"))
    fig = px.bar(errors_df, x='models', y='MSE',
                 labels={'models': 'models', 'MSE': "model's error"}, color="models",
                 title="Model's error for each regression", text_auto=True)
    fig.show()

    return min(min_mse), models[np.argmin(min_mse)]


def choose_reg_model(data, labels):
    X_train, X_test, y_train, y_test = split_data(data, labels)
    num_of_lams = 50
    cv = 5
    ridge_lam, lasso_lam = erm_regressor(X_train, y_train, X_test, y_test, num_of_lams, cv)
    check_simple_regressors(X_train, y_train, X_test, y_test, ridge_lam, lasso_lam, cv)


def split_data(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.2)
    data_with_lbl = np.concatenate([X_train, y_train], axis=1)
    subsets = np.array_split(data_with_lbl, 2)
    X = [sett[:, :-1] for sett in subsets]
    y = [sett[:, -1] for sett in subsets]
    X_train_1, X_train_2, y_train_1, y_train_2 = X[0], X[1], y[0], y[1]
    return X_train, X_test, y_train, y_test


def predict_tumor_size(model):
    X_test = pd.read_csv("test.feats.csv", low_memory=False)
    test_processed = preprocess_data(X_test)
    y_pred = model.predict(test_processed)
    y_pred = pd.DataFrame(y_pred)
    y_pred["Tumor size"] = y_pred[0].apply(lambda x: max(0, x))
    y_pred = y_pred.drop(y_pred.columns[0], axis=1)
    y_pred.to_csv("2.csv", index=False)


def choose_classifying_model(X_train, y_train, X_test, y_test):
    weak_learners = [
        RandomForestClassifier(),
        DecisionTreeClassifier(),
        KNeighborsClassifier()
    ]

    for base_learner in weak_learners:
        f1_micro, f1_macro = train_multioutput_classifier(base_learner,
                                                          X_train, y_train,
                                                          X_test, y_test)
        print(f"Base Learner: {base_learner.__class__.__name__}")
        print("F1 micro:", f1_micro)
        print("F1 macro:", f1_macro)


def train_multioutput_classifier(base_learner, X_train, y_train, X_test,
                                 y_test):
    mlb = MultiLabelBinarizer()

    # Fit and transform the labels for training data
    train_title = y_train.columns[0]
    train_labels = [eval(val) for val in y_train[train_title]]
    mlb.fit(train_labels)
    train_binary = mlb.transform(train_labels)

    # Train the multi-output classifier using the specified base learner
    multi_classifier = MultiOutputClassifier(base_learner).fit(X_train,
                                                               train_binary)

    # Predict the labels for the test data
    multi_classifier_pred = multi_classifier.predict(X_test)

    # Transform the true labels of the test data to the right format
    test_title = y_test.columns[0]
    test_labels = [eval(val) for val in y_test[test_title]]
    test_binary = mlb.transform(test_labels)

    # Calculate F1 scores
    f1_micro = f1_score(test_binary, multi_classifier_pred,
                        average='micro')
    f1_macro = f1_score(test_binary, multi_classifier_pred,
                        average='macro')

    # Return the F1 scores
    return f1_micro, f1_macro


def create_multiclassification_model(base_learner, X, y):
    y = pd.DataFrame(y)
    mlb = MultiLabelBinarizer()

    # Fit and transform the labels for training data
    train_title = y.columns[0]
    train_labels = [eval(val) for val in y[train_title]]
    mlb.fit(train_labels)
    train_binary = mlb.transform(train_labels)

    # Train the multi-output classifier using the specified base learner
    multi_classifier = MultiOutputClassifier(base_learner).fit(X, train_binary)
    return multi_classifier, mlb


def predict_tumor_location(multi_classifier, mlb_model):
    X_test = pd.read_csv("test.feats.csv", low_memory=False)
    X_test_processed = preprocess_data(X_test)
    y_pred = multi_classifier.predict(X_test_processed)
    # Transform binary predictions back to the original representation
    original_predictions = mlb_model.inverse_transform(
        y_pred)
    original_predictions = pd.DataFrame(original_predictions)
    original_predictions["Location of distal metastases"] = original_predictions.apply(
        lambda row: [value for value in [row[0], row[1], row[2]] if
                     value is not None], axis=1)
    original_predictions = original_predictions.drop(
        original_predictions.columns[0:3], axis=1)
    original_predictions.to_csv("1.csv", index=False)


def data_exploration(data):
    pca_data, cluster_labels = perform_pca(data)
    plot_2d_data(pca_data, cluster_labels)
    plot_eigenvalues(pca_data, max_components=2)
    perform_clustering_and_plot(data)


def main():
    # split data into train and test:
    np.random.seed(0)
    X1 = pd.read_csv("train.feats.csv", low_memory=False)
    X2 = pd.read_csv("train.feats.csv", low_memory=False)

    # Task 1 - Classification Model
    y1 = pd.read_csv("train.labels.0.csv")
    base_learner = RandomForestClassifier()
    X1_processed, y1_processed = preprocess_data(X1, y1)
    classification_model, mlb_model = create_multiclassification_model(
        base_learner, X1_processed, y1_processed)
    predict_tumor_location(classification_model, mlb_model)

    # Task 2 - Regression Model
    y2 = pd.read_csv("train.labels.1.csv")
    X2_processed, y2_processed = preprocess_data(X2, y2)
    reg_model = RandomForestRegressor().fit(X2_processed, y2_processed)
    predict_tumor_size(reg_model)


if __name__ == '__main__':
    main()
