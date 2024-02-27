import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data dictionary
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labels
data = data_dict['data']
labels = data_dict['labels']

# Pad or truncate sequences to a fixed length
max_seq_length = max(len(seq) for seq in data)
data = [seq[:max_seq_length] + [0]*(max_seq_length - len(seq)) for seq in data]

# Convert lists to numpy arrays
data = np.asarray(data)
labels = np.asarray(labels)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
