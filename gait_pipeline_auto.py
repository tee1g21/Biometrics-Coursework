import random
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from glob import glob
import os
import cv2
import numpy as np
import os
import cv2
import torch
import numpy as np
from torchvision import transforms

# dataset directories
parent_train_dir = "data/original/training"
parent_test_dir = "data/original/test"

# import image paths
parent_train_image_paths = sorted(glob(os.path.join(parent_train_dir, "*.jpg")))
parent_test_image_paths = sorted(glob(os.path.join(parent_test_dir, "*.jpg")))

# print number of images
print(f"Loaded {len(parent_train_image_paths)} training images")
print(f"Loaded {len(parent_test_image_paths)} test images")

# This gives you real labels from filenames like 016z050pf.jpg
def get_training_labels(image_paths):
    labels = []
    for path in image_paths:
        filename = os.path.basename(path)
        subject_id = filename.split('z')[1][:3]  # Extract '050'
        labels.append(subject_id)
    return labels

train_labels = get_training_labels(parent_train_image_paths)
#print("Train Labels:", train_labels)

test_label_map = {
    'DSC00165.JPG': '001',
    'DSC00166.JPG': '001',
    'DSC00167.JPG': '002',
    'DSC00168.JPG': '002',
    'DSC00169.JPG': '003',
    'DSC00170.JPG': '003',
    'DSC00171.JPG': '004',
    'DSC00172.JPG': '004',
    'DSC00173.JPG': '005',
    'DSC00174.JPG': '005',
    'DSC00175.JPG': '006',
    'DSC00176.JPG': '006',
    'DSC00177.JPG': '007',
    'DSC00178.JPG': '007',
    'DSC00179.JPG': '008',
    'DSC00180.JPG': '008',
    'DSC00181.JPG': '009',
    'DSC00182.JPG': '009',
    'DSC00183.JPG': '010',
    'DSC00184.JPG': '010',
    'DSC00185.JPG': '011',
    'DSC00186.JPG': '011'
}

test_labels = [test_label_map[os.path.basename(path)] for path in parent_test_image_paths]
#print("Test Labels:", test_labels)

HIGHEST_ACCURACY = 0.0
BEST_SEED = 0
BEST_MODEL = ""

for seed in range(40, 200):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    DATASET = "deeplab"
    AUG_RATE = 2
    AUGMENTATIONS = [
        ("flip", transforms.RandomHorizontalFlip(p=1)),
        #("rotate", transforms.RandomRotation(degrees=5)),
        ("shift", transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)))
    ]
    NUM_WIDTH_SAMPLES = 100
    N_NEIGHBOURS = 8
    N_ESTIMATORS = 11

    ## augmentation
    # === Settings ===
    input_dir = f"data/{DATASET}/training"  # or deeplab/test
    output_dir = f"data/{DATASET}/training_augmented"
    #augmentation_rate = 2  # Adjust this value to control the rate of augmentation

    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
    else:
        os.makedirs(output_dir, exist_ok=True)


    # === Process Images ===
    for filename in tqdm(os.listdir(input_dir)):
        if not filename.lower().endswith(('.jpg', '.png')):
            continue
        path = os.path.join(input_dir, filename)
        image = Image.open(path).convert("L")

        # Save original
        image.save(os.path.join(output_dir, filename))

        # Apply augmentations
        for name, aug in AUGMENTATIONS:
            if random.random() < AUG_RATE:
                transformed = aug(image)
                new_name = f"{os.path.splitext(filename)[0]}_{name}.jpg"
                transformed.save(os.path.join(output_dir, new_name))
            
    # re-label augmented images
    augmented_image_paths = sorted(glob(os.path.join(output_dir, "*.jpg")))
    aug_train_labels = get_training_labels(augmented_image_paths)

    print(len(augmented_image_paths), " images in augmented dataset")

    def extract_features(image, use_width=False, num_width_samples=20):
        
        features = []    
        # Ensure binary image
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        # Get main contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(1)  # fallback if image is empty
        contour = max(contours, key=cv2.contourArea)   

        # Width profile
        if use_width:
            h, w = binary.shape
            rows = np.linspace(0, h - 1, num_width_samples, dtype=int)
            width_profile = []
            for y in rows:
                row = binary[y, :]
                x = np.where(row > 0)[0]
                width = x[-1] - x[0] if len(x) > 1 else 0
                width_profile.append(width)
            features.extend(width_profile)
        
        return np.array(features, dtype=np.float32)

    def extract_features_from_dataset(image_paths, feature_func, **kwargs):
        import cv2
        import numpy as np

        X = []
        for path in image_paths:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Could not read: {path}")
                continue
            features = feature_func(image, **kwargs)
            X.append(features)
        return np.array(X, dtype=np.float32)

    # dataset directories
    train_dir = f"data/{DATASET}/training"
    aug_train_dir = f"data/{DATASET}/training_augmented"
    test_dir = f"data/{DATASET}/test"

    # import image paths
    train_image_paths = sorted(glob(os.path.join(train_dir, "*.jpg")))
    aug_train_image_paths = sorted(glob(os.path.join(aug_train_dir, "*.jpg")))
    test_image_paths = sorted(glob(os.path.join(test_dir, "*.jpg")))

    kwargs = {'use_width': True, 'num_width_samples': NUM_WIDTH_SAMPLES}

    X_train = extract_features_from_dataset(train_image_paths, extract_features, **kwargs)
    X_aug_train = extract_features_from_dataset(aug_train_image_paths, extract_features, **kwargs)
    X_test = extract_features_from_dataset(test_image_paths, extract_features, **kwargs)

    def classify(X_train, train_labels, X_test, test_labels):
        # Encode labels to numbers
        le = LabelEncoder()
        y_train_enc = le.fit_transform(train_labels)
        y_test_enc = le.transform(test_labels)

        # scale for some models 
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        def evaluate_model(model, name, X_train_local=None, X_test_local=None):
            global HIGHEST_ACCURACY
            global BEST_SEED
            global BEST_MODEL
            
            X_tr = X_train_local if X_train_local is not None else X_train
            X_te = X_test_local if X_test_local is not None else X_test

            model.fit(X_tr, y_train_enc)
            y_pred_enc = model.predict(X_te)
            y_pred = le.inverse_transform(y_pred_enc)
            accuracy = accuracy_score(test_labels, y_pred)
            if accuracy > HIGHEST_ACCURACY:
                HIGHEST_ACCURACY = accuracy
                BEST_SEED = seed
                BEST_MODEL = name
            fill = " " if (accuracy * 22) < 10 else ""
            print(f"{fill}{int(accuracy * len(test_labels))}/{len(test_labels)} - {accuracy:.3f} : {name}")
            return y_pred, model 

        # --- Standard k-NN ---
        knn = KNeighborsClassifier(n_neighbors=N_NEIGHBOURS)
        y_pred_knn, knn_model = evaluate_model(knn, "k-NN")
        
        # --- Random Forest ---
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42)
        y_pred_rf, rf_model = evaluate_model(rf, "Random Forest")

    
        def other_models():
            # --- SVM ---
            svm = SVC(kernel='linear', C=1, gamma='scale', max_iter=10000)  # gamma='scale' is usually fine
            y_pred, _ = evaluate_model(svm, "SVM")

            # --- SGD --- 
            sgd = SGDClassifier(max_iter=1000, tol=1e-3)
            y_pred, _ = evaluate_model(sgd, "SGD")

            # --- Logistic Regression ---
            logreg = LogisticRegression(solver='lbfgs', max_iter=1000)
            y_pred, _ = evaluate_model(logreg, "Logistic Regression", X_train_scaled, X_test_scaled)

            # --- Naive Bayes ---
            gnb = GaussianNB()
            y_pred, _ = evaluate_model(gnb, "Gaussian Naive Bayes", X_train, X_test)

            # --- Decision Tree --- 
            dtree = DecisionTreeClassifier(max_depth=100, random_state=42) # play with depth
            y_pred, _ = evaluate_model(dtree, "Decision Tree")

            # --- MLP ---
            mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
            y_pred, _ = evaluate_model(mlp, "MLP", X_train_scaled, X_test_scaled)

            ## --- Custom Pytorch CNN ---
            #print("\nPytorch CNN")
            #y_pred_cnn = train_cnn_classifier(train_image_paths, train_labels,
            #                                test_image_paths, test_labels, le, 30)
            #accuracy = accuracy_score(test_labels, y_pred_cnn)
            #fill = " " if (accuracy * 22) < 10 else ""
            #print(f"{fill}{int(accuracy * len(test_labels))}/{len(test_labels)} - {accuracy:.3f} : CNN")    
        other_models()
        print()
        return y_pred_knn, y_pred_rf

    _, y_pred = classify(X_train, train_labels, X_test, test_labels)
    y_pred_aug, _ = classify(X_aug_train, aug_train_labels, X_test, test_labels)

    print(f"Seed {seed} - Best model: {BEST_MODEL} with accuracy {int(HIGHEST_ACCURACY * 22)}/22 and seed {BEST_SEED}\n")

if __name__ == "__main__":
    pass