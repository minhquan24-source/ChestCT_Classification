from collections import Counter
import numpy as np
import os

# ZeroR Model
class ZeroR:
    def fit(self, labels):
        # Find the most frequent class in the training labels
        most_common = Counter(labels).most_common(1)
        self.predicted_class = most_common[0][0] if most_common else None

    def predict(self, data):
        # Predict the most frequent class for any input
        return [self.predicted_class] * len(data)

if __name__ == '__main__':
    train_dir = './data/curated/train'

    classes = os.listdir(train_dir)
    if '.DS_Store' in classes:
        classes.remove('.DS_Store')
    train_files = []

    for i, c in enumerate(classes):
        images = os.listdir(os.path.join(train_dir, c))
        train_files.extend([(os.path.join(train_dir, c, img), i) for img in images])

    validation_dir = './data/curated/valid'
    val_files = []

    for i, c in enumerate(classes):
        images = os.listdir(os.path.join(validation_dir, c))
        val_files.extend([(os.path.join(validation_dir, c, img), i) for img in images])

    # Load training labels to fit ZeroR
    train_labels = [label for _, label in train_files]

    # Initialize and fit the ZeroR model
    zero_r = ZeroR()
    zero_r.fit(train_labels)

    # Use ZeroR to predict validation data
    zero_r_predictions = zero_r.predict(val_files)

    # Calculate ZeroR accuracy
    zero_r_correct_predictions = sum(p == t for p, (_, t) in zip(zero_r_predictions, val_files))
    zero_r_total_predictions = len(val_files)
    zero_r_accuracy = zero_r_correct_predictions / zero_r_total_predictions

    print(f'ZeroR Baseline Accuracy: {zero_r_accuracy * 100:.2f}%')
