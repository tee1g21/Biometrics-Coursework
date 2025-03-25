## import dataset
from glob import glob
import os

# dataset directories
parent_train_dir = "data/original/training"
parent_test_dir = "data/original/test"

# import image paths
parent_train_image_paths = sorted(glob(os.path.join(parent_train_dir, "*.jpg")))
parent_test_image_paths = sorted(glob(os.path.join(parent_test_dir, "*.jpg")))

# print number of images
print(f"Loaded {len(parent_train_image_paths)} training images")
print(f"Loaded {len(parent_test_image_paths)} test images")

# ============================================================

# label data
def get_training_labels(image_paths):
    labels = []
    for path in image_paths:
        filename = os.path.basename(path)
        subject_id = filename.split('z')[1][:3]  # Extract '050'
        labels.append(subject_id)
    return labels

train_labels = get_training_labels(parent_train_image_paths)
print("Train Labels:", train_labels)

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
print("Test Labels:", test_labels)

# ============================================================

# feature extraction
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

import cv2
import numpy as np
from scipy.fft import fft
from pyefd import elliptic_fourier_descriptors
from skimage.morphology import skeletonize


def extract_sil_features(image, 
                     use_hu=False,
                     use_fourier=False, num_fourier=10, 
                     use_width=False, num_width_samples=20,
                     use_height=False, num_height_samples=20,
                     use_area=False, use_perimeter=False, use_compactness=False, use_dispersion=False,
                     use_efd=False, num_efd=10, 
                     use_aspect=False, 
                        use_skeleton_length=False
                     ):
    
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
        
    if use_area:
        area = cv2.contourArea(contour)
        features.append(area)
    
    # perimeter
    if use_perimeter:
        perimeter = cv2.arcLength(contour, closed=True)
        features.append(perimeter)
    
    # compactness
    if use_compactness:
        compactness = (4 * np.pi * area) / (perimeter ** 2 + 1e-5)
        features.append(compactness)

    #dispersion
    if use_dispersion:
        M = cv2.moments(contour)
        cx = int(M["m10"] / (M["m00"] + 1e-5))
        cy = int(M["m01"] / (M["m00"] + 1e-5))
        dispersion = max(np.sqrt((pt[0][0] - cx)**2 + (pt[0][1] - cy)**2) for pt in contour)
        features.append(dispersion)

    # eliptical fourier Descriptors
    if use_efd:
        # Contour must be 2D float32
        cnt = contour.squeeze().astype(np.float32)
        if cnt.ndim < 2:
            cnt = cnt[np.newaxis, :]  # safety

        coeffs = elliptic_fourier_descriptors(cnt, order=num_efd, normalize=True)
        coeffs = coeffs.flatten()[1:]  # Remove first 4 (translation)
        features.extend(coeffs)
    
    # hu moments 
    if use_hu:
        moments = cv2.moments(binary)
        hu = cv2.HuMoments(moments).flatten()
        features.extend(hu)

    # height profile
    if use_height:
        h, w = binary.shape
        cols = np.linspace(0, w - 1, num_height_samples, dtype=int)
        height_profile = []
        for x in cols:
            col = binary[:, x]
            y = np.where(col > 0)[0]
            height = y[-1] - y[0] if len(y) > 1 else 0
            height_profile.append(height)
        features.extend(height_profile)

    # Aspect ratio
    if use_aspect:
        x, y, w, h = cv2.boundingRect(contour)
        aspect = h / (w + 1e-5)
        features.append(aspect)

    # Total skeleton length
    if use_skeleton_length:
        bin_img = (binary > 0).astype(np.uint8)
        skeleton = skeletonize(bin_img).astype(np.uint8)
        length = np.count_nonzero(skeleton)
        features.append(length)
        
    # Fourier Descriptors
    if use_fourier:
        cnt = contour.squeeze()
        if len(cnt.shape) == 1:  # safety check
            cnt = cnt[np.newaxis, :]
        complex_cnt = cnt[:, 0] + 1j * cnt[:, 1]
        fourier = fft(complex_cnt)
        desc = np.abs(fourier[:num_fourier])
        features.extend(desc)

    return np.array(features, dtype=np.float32)

import mediapipe as mp
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(static_image_mode=True)

def extract_pose_ratios(
    keypoints,
    use_foot_to_lower_leg=False,
    use_lower_to_upper_leg=False,
    use_leg_to_arm=False,
    use_upper_arm_to_leg=False,
    use_shoulder_to_hip_width=False,
    use_shoulder_to_upper_leg=False,
    use_head_to_torso=False
):
    """
    Compute selected ratios from keypoints using 3D distances.
    Each ratio is included only if its toggle is set to True.
    """
    def dist(a, b):
        return np.linalg.norm(np.array(keypoints[a]) - np.array(keypoints[b]))

    def avg_dist(a1, b1, a2, b2):
        return (dist(a1, b1) + dist(a2, b2)) / 2

    features = []

    # Leg lengths
    lower_leg = avg_dist(25, 27, 26, 28)  # knee → ankle
    upper_leg = avg_dist(23, 25, 24, 26)  # hip → knee
    leg_length = lower_leg + upper_leg

    # Arm lengths
    lower_arm = avg_dist(13, 15, 14, 16)  # elbow → wrist
    upper_arm = avg_dist(11, 13, 12, 14)  # shoulder → elbow
    arm_length = lower_arm + upper_arm

    # Hip and shoulder widths
    shoulder_width = dist(11, 12)
    hip_width = dist(23, 24)

    # Head and torso
    nose = np.array(keypoints[0])
    shoulders_mid = (np.array(keypoints[11]) + np.array(keypoints[12])) / 2
    hips_mid = (np.array(keypoints[23]) + np.array(keypoints[24])) / 2
    head_height = np.linalg.norm(nose - shoulders_mid)
    torso_height = np.linalg.norm(shoulders_mid - hips_mid)

    # Foot length
    foot_length = avg_dist(29, 31, 30, 32)  # heel → toe

    if use_foot_to_lower_leg:
        features.append(foot_length / (lower_leg + 1e-5))
    if use_lower_to_upper_leg:
        features.append(lower_leg / (upper_leg + 1e-5))
    if use_leg_to_arm:
        features.append(leg_length / (arm_length + 1e-5))
    if use_upper_arm_to_leg:
        features.append(upper_arm / (upper_leg + 1e-5))
    if use_shoulder_to_hip_width:
        features.append(shoulder_width / (hip_width + 1e-5))
    if use_shoulder_to_upper_leg:
        features.append(shoulder_width / (upper_leg + 1e-5))
    if use_head_to_torso:
        features.append(head_height / (torso_height + 1e-5))

    return np.array(features, dtype=np.float32)


def extract_pose_features(image, mode=None, use_visibility=False, use_ratios=False, **ratio_kwargs):
    
    features = []

    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    results = pose_model.process(image)
    if not results.pose_landmarks:
        base_size = 0
        if mode == 'xy' or mode == 'xy+xyz':
            base_size += 66
        if mode == 'xyz' or mode == 'xy+xyz':
            base_size += 99
        if use_visibility:
            base_size += 33
        if use_ratios:
            base_size += sum(ratio_kwargs.values())
        return np.zeros(base_size, dtype=np.float32)

    keypoints = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]

    if mode == 'xyz' or mode == 'xy+xyz':
        features.extend([coord for p in keypoints for coord in p])
    if mode == 'xy' or mode == 'xy+xyz':
        features.extend([coord for p in keypoints for coord in p[:2]])

    if use_visibility:
        features.extend([lm.visibility for lm in results.pose_landmarks.landmark])

    if use_ratios:
        ratio_features = extract_pose_ratios(keypoints, **ratio_kwargs)
        features.extend(ratio_features)

    return np.array(features, dtype=np.float32)

# ============================================================

# classification

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

def classify(X_train, train_labels, X_test, test_labels, n_neighbours, n_estimators, tree_depth=5, seed=42):
    # Encode labels to numbers
    le = LabelEncoder()
    y_train_enc = le.fit_transform(train_labels)
    y_test_enc = le.transform(test_labels)

    # scale for some models 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    def evaluate_model(model, name, X_train_local=None, X_test_local=None):
        X_tr = X_train_local if X_train_local is not None else X_train
        X_te = X_test_local if X_test_local is not None else X_test

        model.fit(X_tr, y_train_enc)
        y_pred_enc = model.predict(X_te)
        y_pred = le.inverse_transform(y_pred_enc)
        accuracy = accuracy_score(test_labels, y_pred)
        fill = " " if (accuracy * 22) < 10 else ""
        print(f"{fill}{int(accuracy * len(test_labels))}/{len(test_labels)} - {accuracy:.3f} : {name}")
        # append feature vector and accuracy to csv 
        with open('results.txt', 'a') as f:
            f.write(f"{name},{int(accuracy * 22)}\n")
        
        return y_pred, model 

    # --- Standard k-NN ---
    knn = KNeighborsClassifier(n_neighbors=n_neighbours)
    y_pred_knn, knn_model = evaluate_model(knn, "k-NN")
    
    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    y_pred_rf, rf_model = evaluate_model(rf, "Random Forest")

     # --- Logistic Regression ---
    logreg = LogisticRegression(solver='liblinear', max_iter=1000)
    y_pred_lr, _ = evaluate_model(logreg, "Logistic Regression", X_train_scaled, X_test_scaled)
  
    # --- SVM ---
    svm = SVC(kernel='linear', C=1, gamma='scale', max_iter=100000)  # gamma='scale' is usually fine
    y_pred_svm, _ = evaluate_model(svm, "SVM")

    # --- Decision Tree --- 
    dtree = DecisionTreeClassifier(max_depth=tree_depth, random_state=42) # play with depth
    y_pred_dtree, _ = evaluate_model(dtree, "Decision Tree")
  
    def other_models():
        
        # --- SGD --- 
        sgd = SGDClassifier(max_iter=1000, tol=1e-3)
        #y_pred, _ = evaluate_model(sgd, "SGD")  

        # --- Naive Bayes ---
        gnb = GaussianNB()
        #y_pred, _ = evaluate_model(gnb, "Gaussian Naive Bayes", X_train, X_test)

                # --- MLP ---
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        #y_pred, _ = evaluate_model(mlp, "MLP", X_train_scaled, X_test_scaled)

                
    other_models()
    print()
    return [y_pred_knn, y_pred_rf, y_pred_lr, y_pred_svm, y_pred_dtree]


# dataset directories

SIL_DATASET = "deeplab"

# --- silhouette features ---
train_dir = f"data/{SIL_DATASET}/training"
aug_train_dir = f"data/{SIL_DATASET}/training_augmented"
test_dir = f"data/{SIL_DATASET}/test"

# import image paths
train_image_paths = sorted(glob(os.path.join(train_dir, "*.jpg")))
aug_train_image_paths = sorted(glob(os.path.join(aug_train_dir, "*.jpg")))
test_image_paths = sorted(glob(os.path.join(test_dir, "*.jpg")))

import itertools
from datetime import datetime

widths = [50]
binary_sil_options = {
    'use_efd': [True, False],
    'use_aspect': [True, False],
    'use_skeleton_length': [True, False],
}
pose_modes = ['none', 'xy', 'xyz', 'xyv']
vis_flags = [True, False]
ratio_flags = [True, False]

# All combinations of binary silhouette flags
sil_flag_combos = list(itertools.product(*binary_sil_options.values()))

# Open results file once
results_file = open("results.txt", "a")

for width in widths:
    for sil_flags in sil_flag_combos:
        for pose_mode in pose_modes:
            for vis in vis_flags:
                for ratio in ratio_flags:
                    try:
                        swargs = {
                            'use_width': True,
                            'num_width_samples': width,
                            'use_efd': sil_flags[0],
                            'use_aspect': sil_flags[1],
                            'use_skeleton_length': sil_flags[2],
                        }

                        pwargs = {
                            'mode': pose_mode,
                            'use_visibility': vis,
                            'use_ratios': ratio,
                            'use_foot_to_lower_leg': True,
                            'use_lower_to_upper_leg': True,
                            'use_leg_to_arm': True,
                            'use_upper_arm_to_leg': True,
                            'use_shoulder_to_hip_width': True,
                            'use_shoulder_to_upper_leg': True,
                            'use_head_to_torso': True,
                        }

                        X_train_sil = extract_features_from_dataset(train_image_paths, extract_sil_features, **swargs)
                        X_test_sil = extract_features_from_dataset(test_image_paths, extract_sil_features, **swargs)

                        X_train_pose = extract_features_from_dataset(parent_train_image_paths, extract_pose_features, **pwargs)
                        X_test_pose = extract_features_from_dataset(parent_test_image_paths, extract_pose_features, **pwargs)

                        X_train_combined = np.hstack((X_train_sil, X_train_pose))
                        X_test_combined = np.hstack((X_test_sil, X_test_pose))

                        


                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Save to file
                        results_file.write(f"[{timestamp}] | width={width}, mode={pose_mode}, vis={vis}, ratio={ratio}, swargs={swargs}, pwargs={pwargs}\n")
                        
                        y_pred_sil = classify(X_train_sil, train_labels, X_test_sil, test_labels,
                                             n_neighbours=5, n_estimators=120, tree_depth=6, seed=42)
                        
                        y_pred = classify(X_train_combined, train_labels, X_test_combined, test_labels,
                                             n_neighbours=5, n_estimators=110, tree_depth=6, seed=42)
                        
                        results_file.flush()

                    except Exception as e:
                        results_file.write(f"[ERROR] width={width}, swargs={swargs}, pwargs={pwargs} | {e}\n")
                        results_file.flush()

results_file.close()
