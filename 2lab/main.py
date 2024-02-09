import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        ''' constructor '''

        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree_Classifcator:
    def __init__(self, min_samples_split, max_depth):
        ''' constructor '''
        # initialize the root of the tree
        self.root = None
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.HEADER = None


    def build_tree(self, df, curr_depth=0, header=None):
        ''' recursive function to build the tree '''

        X, Y = df[:, :-1], df[:, -1]
        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(df, num_samples, num_features)
            # check if information gain is positive and feature index is not None
            if best_split.get("feature_index") is not None and best_split["info_gain"] > 0:
                # recur left
                left_subtree = self.build_tree(best_split["df_left"], curr_depth+1, header=header)
                # recur right
                right_subtree = self.build_tree(best_split["df_right"], curr_depth+1, header=header)
                # return decision node
                return Node(
                    best_split["feature_index"], 
                    best_split["threshold"], 
                    left_subtree, 
                    right_subtree, 
                    best_split["info_gain"]
                )

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)



    def get_best_split(self, df, num_samples, num_features):
        ''' function to find the best split '''
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            if feature_index not in set(self.HEADER):
                continue  # Skip features not present in the header

            feature_values = df[:, feature_index]
            # Convert feature values to strings to handle both numeric and categorical features
            feature_values_str = feature_values.astype(str)
            possible_threshold = np.unique(feature_values_str)

            # loop over all the feature values present in the area
            for threshold in possible_threshold:
                # get current split
                df_left, df_right = self.split(df, feature_index, threshold)
                # check if childs are not null
                if len(df_left) > 0 and len(df_right) > 0:
                    y, left_y, right_y = df[:, -1], df_left[:, -1], df_right[:, -1]
                    # compute information again
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["df_left"] = df_left
                        best_split["df_right"] = df_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # return best split
        return best_split


    def split(self, df, feature_index, threshold):
        ''' function to split the data '''
        feature_values = df[:, feature_index]

        if np.issubdtype(feature_values.dtype, np.number):
            # Numeric feature
            df_left = df[feature_values <= threshold]
            df_right = df[feature_values > threshold]
        else:
            # Categorical feature
            df_left = df[feature_values == threshold]
            df_right = df[feature_values != threshold]

        return df_left, df_right
    
    def entropy(self, y):
        ''' function to compute entropy '''

        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = np.sum(y == cls) / len(y)  # Use sum directly for count
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    

    def gini_index(self, y):
        ''' function to compute gini index '''

        y = np.asarray(y)

        if y.dtype == object:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = np.sum(y == cls) / len(y)
            gini += p_cls**2
        return 1 - gini


    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        # Convert 'Y' to a consistent type (float) if possible
        try:
            Y = np.asarray(Y, dtype=float)
        except ValueError:
            # Handle non-numeric values differently (e.g., return a default value)
            return self.handle_non_numeric(Y)

        # Handle NaN values in Y by replacing with the mean value
        Y = np.nan_to_num(Y, nan=np.nanmean(Y))

        unique, counts = np.unique(Y, return_counts=True)
        return unique[np.argmax(counts)]

    def handle_non_numeric(self, Y):
        ''' function to handle non-numeric values in leaf node '''
        # Replace NaN values with a default value
        Y = np.nan_to_num(Y, nan=0.0)
        numeric_values = [value for value in Y if isinstance(value, (int, float))]
        if not numeric_values:
            # Handle the case where there are no numeric values
            return 0.0  # or any default value you choose
        imputed_value = np.nanmean(numeric_values)
        return imputed_value
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        if self.HEADER is None:
            self.HEADER = list(X.columns)

        if isinstance(Y, pd.DataFrame):
            df = pd.concat([X, Y], axis=1)
            label_encoder = LabelEncoder()
            Y_encoded = label_encoder.fit_transform(Y.ravel())
        else:
            df = pd.concat([X, pd.DataFrame(Y, columns=[self.HEADER[-1]])], axis=1)
            label_encoder = LabelEncoder()
            Y_encoded = label_encoder.fit_transform(Y.ravel())

        self.root = self.build_tree(df.values, curr_depth=0, header=self.HEADER)
        self.label_encoder = label_encoder


    

    def make_prediction(self, x, tree, header):
        ''' function to predict a single data point '''
        feature_index = tree.feature_index

        if feature_index is None or feature_index not in header:
            print(f"Skipping prediction for this branch. Feature index: {feature_index}")
            return None

        feature_val = x[header.index(feature_index)]

        if feature_val is None:
            print(f"Feature value is None for feature index {feature_index}. Skipping prediction.")
            return None

        # Convert feature_val to a string if it's not already
        feature_val = str(feature_val)

        if feature_val == tree.threshold:
            print(f"Going left for feature index {feature_index}, threshold: {tree.threshold}")
            return self.make_prediction(x, tree.left, header)
        else:
            print(f"Going right for feature index {feature_index}, threshold: {tree.threshold}")
            return self.make_prediction(x, tree.right, header)


    def predict(self, X):
            ''' function to predict new dataset '''
            predictions = [self.make_prediction(x, self.root, header=self.HEADER) for x in X.to_numpy()]

            return predictions
    
    def handle_nan_value(self, nan_value):
        ''' function to handle NaN values in predictions '''
        # Replace NaN values with the majority class in the training data
        return self.calculate_leaf_value(self.label_encoder.classes_)
    
def model_pred(train_X, train_y, test_X, test_y):
    pass

def main():
    HEADER = list(pd.read_csv("data/грибы.txt", sep="\t", nrows=1, encoding="Windows-1251").columns)

    df = pd.read_csv("data/грибы.txt", skiprows=1, header=None, names=HEADER, encoding="Windows-1251", sep="\t")
    df.dropna(inplace=True)

    # Identify categorical features and apply one-hot encoding
    X = pd.get_dummies(df[HEADER[1:]])  # Exclude the target column from one-hot encoding
    y = df[HEADER[0]].values  # Assuming the target column is the first one in HEADER

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)

    # Use the decision tree classifier from your implementation
    tree_orig = DecisionTreeClassifier()
    custom_f1_score, predictions = model_pred(train_X, train_y, test_X, test_y)

    # Fit the decision tree on the preprocessed data
    tree_orig.fit(train_X, train_y)
    pred_tree_orig = tree_orig.predict(test_X)
    f1_orig = f1_score(test_y, pred_tree_orig, pos_label="съедобный")
    np.set_printoptions(threshold=np.inf)

    print("F1 Score (Custom Decision Tree):", custom_f1_score)
    print("Predictions (Custom Decision Tree):", predictions)
    print("F1 Score (Original Decision Tree):", f1_orig)
    print("Predictions (Original Decision Tree):", pred_tree_orig)

if __name__ == "__main__":
    main()
