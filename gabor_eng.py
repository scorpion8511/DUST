# import numpy as np
# import torch
# from scipy.linalg import inv
# from numpy.linalg import eigh
# from skimage.filters import gabor



# # Utility Function to Convert NumPy Arrays to Torch Tensors
# def to_torch(ndarray):
#     from collections.abc import Sequence
#     if ndarray is None:
#         return None
#     if isinstance(ndarray, Sequence):
#         return [to_torch(ndarray_) for ndarray_ in ndarray if ndarray_ is not None]
#     if type(ndarray).__module__ == 'numpy':
#         return torch.from_numpy(ndarray)
#     if torch.is_tensor(ndarray):
#         return ndarray
#     raise ValueError('Fail to convert')

# # LDA Class
# class LDA:
#     def __init__(self, shrinkage=None, priors=None, n_components=None):
#         self.shrinkage = shrinkage
#         self.priors = priors
#         self.n_components = n_components

#     def _cov(self, X, shrinkage=-1):
#         emp_cov = np.cov(np.asarray(X).T, bias=1)
#         if shrinkage < 0:
#             return emp_cov
#         n_features = emp_cov.shape[0]
#         mu = np.trace(emp_cov) / n_features
#         shrunk_cov = (1.0 - shrinkage) * emp_cov
#         shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
#         return shrunk_cov

#     def softmax(self, X, copy=True):
#         if copy:
#             X = np.copy(X)
#         max_prob = np.max(X, axis=1).reshape((-1, 1))
#         X -= max_prob
#         np.exp(X, X)
#         sum_prob = np.sum(X, axis=1).reshape((-1, 1))
#         X /= sum_prob
#         return X

#     def fit(self, X, y):
#         self.classes_ = np.unique(y)
#         n_classes = len(self.classes_)

#         max_components = min(len(self.classes_) - 1, X.shape[1])
#         if self.n_components is None:
#             self._max_components = max_components
#         else:
#             if self.n_components > max_components:
#                 raise ValueError(
#                     "n_components cannot be larger than min(n_features, n_classes - 1)."
#                 )
#             self._max_components = self.n_components

#         _, y_t = np.unique(y, return_inverse=True)
#         self.priors_ = np.bincount(y_t) / float(len(y))
#         self._solve_eigen(X, y, shrinkage=self.shrinkage)

#         return self

#     def _solve_eigen(self, X, y, shrinkage):
#         classes, y = np.unique(y, return_inverse=True)
#         cnt = np.bincount(y)

#         means = np.zeros(shape=(len(classes), X.shape[1]))
#         np.add.at(means, y, X)
#         means /= cnt[:, None]
#         self.means_ = means

#         cov = np.zeros(shape=(X.shape[1], X.shape[1]))
#         for idx, group in enumerate(classes):
#             Xg = X[y == group, :]
#             cov += self.priors_[idx] * np.atleast_2d(self._cov(Xg))
#         self.covariance_ = cov

#         Sw = self.covariance_
#         if self.shrinkage is None:
#             shrinkage = 0.1
#         St = self._cov(X, shrinkage=shrinkage)

#         n_features = Sw.shape[0]
#         mu = np.trace(Sw) / n_features
#         shrunk_Sw = (1.0 - shrinkage) * Sw
#         shrunk_Sw.flat[:: n_features + 1] += shrinkage * mu

#         Sb = St - shrunk_Sw

#         evals, evecs = eigh(inv(shrunk_Sw).dot(Sb))
#         evecs = evecs[:, np.argsort(evals)[::-1]]
#         self.scalings_ = evecs
#         self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
#         self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
#             self.priors_
#         )

#     def predict_proba(self, X):
#         logits = np.dot(X, self.coef_.T) + self.intercept_
#         return self.softmax(logits)

# # Energy Score Function
# def Energy_Score(logits, percent, tail):
#     logits = to_torch(logits)
#     energy_score = torch.logsumexp(logits, dim=-1).numpy()
#     if tail == 'bot':
#         chs = list(np.argsort(energy_score)[: int(percent * len(energy_score) // 100)])
#     else:
#         chs = list(np.argsort(energy_score)[-int(percent * len(energy_score) // 100):])
#     energy_score = energy_score[chs].mean()
#     return energy_score

# # Workflow for LDA and Energy Score Computation
# def compute_scores(test_features_path):
#     # Load test features and labels
#     test_data = torch.load(test_features_path)

#     # Handle differences in data format
#     def to_numpy(data):
#         if isinstance(data, torch.Tensor):
#             return data.numpy()
#         elif isinstance(data, np.ndarray):
#             return data
#         else:
#             raise ValueError("Unsupported data type")

#     test_feats_np = to_numpy(test_data['embeddings'])
#     test_labels_np = to_numpy(test_data['labels'])

#     # **LDA Score**
#     lda = LDA(shrinkage=0.1)
#     lda.fit(test_feats_np, test_labels_np)  # Use test data for fitting
#     lda_probabilities = lda.predict_proba(test_feats_np)

#     # Average probability of correct class
#     lda_score = np.sum(lda_probabilities[np.arange(len(test_labels_np)), test_labels_np]) / len(test_labels_np)
#     print(f"LDA Score: {lda_score}")

#     # **Energy Score**
#     logits = np.dot(test_feats_np, lda.coef_.T) + lda.intercept_
#     logits_torch = torch.tensor(logits)

#     # Full Energy Score
#     full_energy_score = Energy_Score(logits_torch, percent=100, tail='bot')
#     print(f"Energy Score (Full): {full_energy_score}")  

# # Function to compute Gabor features for images
# def pad_to_square(array):
#     """
#     Pads a 1D array to the nearest perfect square size.

#     Parameters:
#     - array: 1D NumPy array.

#     Returns:
#     - padded_array: Padded 1D array to make it a perfect square.
#     """
#     size = array.size
#     next_square = int(np.ceil(np.sqrt(size)) ** 2)  # Find the nearest perfect square
#     padded_array = np.zeros(next_square, dtype=array.dtype)  # Create a padded array
#     padded_array[:size] = array  # Copy original array values
#     return padded_array

# def compute_gabor_features(features, frequencies=[0.1, 0.2, 0.3]):
#     """
#     Apply Gabor filters to extract texture features from embeddings.

#     Parameters:
#     - features: NumPy array of 1D image embeddings.
#     - frequencies: List of Gabor filter frequencies.

#     Returns:
#     - gabor_feats: NumPy array of Gabor features for all images.
#     """
#     gabor_feats = []
#     for feature in features:
#         padded_feature = pad_to_square(feature)  # Pad the feature to make it a square
#         dim = int(np.sqrt(padded_feature.size))  # Compute the dimension of the square
#         image = padded_feature.reshape(dim, dim)  # Reshape the padded feature into 2D
#         image_feats = []
#         for freq in frequencies:
#             _, gabor_resp = gabor(image, frequency=freq)
#             image_feats.append(gabor_resp.flatten())  # Flatten the response for each frequency
#         gabor_feats.append(np.concatenate(image_feats))  # Concatenate responses for all frequencies
#     return np.array(gabor_feats)



# # Workflow for LDA and Gabor Feature Score Computation
# def compute_gabor_scores(test_features_path):
#     # Load test features and labels
#     test_data = torch.load(test_features_path)

#     # Handle differences in data format
#     def to_numpy(data):
#         if isinstance(data, torch.Tensor):
#             return data.numpy()
#         elif isinstance(data, np.ndarray):
#             return data
#         else:
#             raise ValueError("Unsupported data type")

#     test_feats_np = to_numpy(test_data['embeddings'])
#     test_labels_np = to_numpy(test_data['labels'])

#     # Compute Gabor features from test embeddings
#     gabor_features = compute_gabor_features(test_feats_np)

#     # **LDA Score on Gabor Features**
#     lda = LDA(shrinkage=0.1)
#     lda.fit(gabor_features, test_labels_np)  # Use Gabor features for fitting
#     lda_probabilities = lda.predict_proba(gabor_features)

#     # Average probability of correct class
#     lda_score = np.sum(lda_probabilities[np.arange(len(test_labels_np)), test_labels_np]) / len(test_labels_np)
#     print(f"Gabor-LDA Score: {lda_score}")

#     # **Energy Score on Gabor Features**
#     logits = np.dot(gabor_features, lda.coef_.T) + lda.intercept_
#     logits_torch = torch.tensor(logits)

#     # Full Energy Score
#     full_energy_score = Energy_Score(logits_torch, percent=100, tail='bot')
#     print(f"Gabor Energy Score (Full): {full_energy_score}")

# # Example Usage
# # test_features_path =  "/home/jovyan/work/tran_est/saved_models_and_features_conch_brk01/conch_ViT-B-16_pretrained_features.pth"
# # test_features_path =  "/home/jovyan/work/tran_est/saved_models_and_features_giga_brk01/giga_model_vit_large_patch16_224_pretrained_features.pth"
# # test_features_path =  "/home/jovyan/work/tran_est/saved_models_and_features_phikon_brk01/phikon_v2_train_features.pth"

# # test_features_path =  "/home/jovyan/work/tran_est/saved_models_and_features_uni_brk01/uni_vit_large_patch16_pretrained_features.pth"

# # test_features_path =  "/home/jovyan/work/PSE_dhmc_kid/dhmc_kid_pth/giga_dhmc_kid.pth"
# uni = "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/uni_dhmc_lung.pth"
# conch= "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/conch_dhmc_lung.pth"
# giga = "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/giga_dhmc_lung.pth"
# phikon= "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/phikon_dhmc_lung.pth"
# virchow = "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/vir_dhmc_lung.pth"





# # test_features_path = "/home/jovyan/work/tran_est/saved_models_and_features_giga0005/giga_model_vit_large_patch16_224_pretrained_features.pth" # giga
# # test_features_path = "/home/jovyan/work/tran_est/saved_models_and_features_phikon0005/phikon_v2_test_features.pth"  # phikon
# # test_features_path = "/home/jovyan/work/tran_est/saved_models_and_features_uni_0005/uni_vit_large_patch16_pretrained_features.pth"  # uni



# compute_gabor_scores(test_features_path)



import numpy as np
import torch
from scipy.linalg import inv
from numpy.linalg import eigh
from skimage.filters import gabor

# Utility Function to Convert NumPy Arrays to Torch Tensors
def to_torch(ndarray):
    from collections.abc import Sequence
    if ndarray is None:
        return None
    if isinstance(ndarray, Sequence):
        return [to_torch(ndarray_) for ndarray_ in ndarray if ndarray_ is not None]
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    if torch.is_tensor(ndarray):
        return ndarray
    raise ValueError('Fail to convert')

# LDA Class
class LDA:
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components

    def _cov(self, X, shrinkage=-1):
        emp_cov = np.cov(np.asarray(X).T, bias=1)
        if shrinkage < 0:
            return emp_cov
        n_features = emp_cov.shape[0]
        mu = np.trace(emp_cov) / n_features
        shrunk_cov = (1.0 - shrinkage) * emp_cov
        shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
        return shrunk_cov

    def softmax(self, X, copy=True):
        if copy:
            X = np.copy(X)
        max_prob = np.max(X, axis=1).reshape((-1, 1))
        X -= max_prob
        np.exp(X, X)
        sum_prob = np.sum(X, axis=1).reshape((-1, 1))
        X /= sum_prob
        return X

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        max_components = min(len(self.classes_) - 1, X.shape[1])
        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        _, y_t = np.unique(y, return_inverse=True)
        self.priors_ = np.bincount(y_t) / float(len(y))
        self._solve_eigen(X, y, shrinkage=self.shrinkage)

        return self

    def _solve_eigen(self, X, y, shrinkage):
        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)

        means = np.zeros(shape=(len(classes), X.shape[1]))
        np.add.at(means, y, X)
        means /= cnt[:, None]
        self.means_ = means

        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(self._cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_
        if self.shrinkage is None:
            shrinkage = 0.1
        St = self._cov(X, shrinkage=shrinkage)

        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += shrinkage * mu

        Sb = St - shrunk_Sw

        evals, evecs = eigh(inv(shrunk_Sw).dot(Sb))
        evecs = evecs[:, np.argsort(evals)[::-1]]
        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def predict_proba(self, X):
        logits = np.dot(X, self.coef_.T) + self.intercept_
        return self.softmax(logits)

# Energy Score Function
def Energy_Score(logits, percent, tail):
    logits = to_torch(logits)
    energy_score = torch.logsumexp(logits, dim=-1).numpy()
    if tail == 'bot':
        chs = list(np.argsort(energy_score)[: int(percent * len(energy_score) // 100)])
    else:
        chs = list(np.argsort(energy_score)[-int(percent * len(energy_score) // 100):])
    energy_score = energy_score[chs].mean()
    return energy_score

# Workflow for LDA and Energy Score Computation
def compute_scores(test_features_path):
    # Load test features and labels
    test_data = torch.load(test_features_path)

    # Handle differences in data format
    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unsupported data type")

    test_feats_np = to_numpy(test_data['embeddings'])
    test_labels_np = to_numpy(test_data['labels'])

    # **LDA Score**
    lda = LDA(shrinkage=0.1)
    lda.fit(test_feats_np, test_labels_np)  # Use test data for fitting
    lda_probabilities = lda.predict_proba(test_feats_np)

    # Average probability of correct class
    lda_score = np.sum(lda_probabilities[np.arange(len(test_labels_np)), test_labels_np]) / len(test_labels_np)
    print(f"LDA Score: {lda_score}")

    # **Energy Score**
    logits = np.dot(test_feats_np, lda.coef_.T) + lda.intercept_
    logits_torch = torch.tensor(logits)

    # Full Energy Score
    full_energy_score = Energy_Score(logits_torch, percent=100, tail='bot')
    print(f"Energy Score (Full): {full_energy_score}")  

# Function to compute Gabor features for images
def pad_to_square(array):
    size = array.size
    next_square = int(np.ceil(np.sqrt(size)) ** 2)
    padded_array = np.zeros(next_square, dtype=array.dtype)
    padded_array[:size] = array
    return padded_array

def compute_gabor_features(features, frequencies=[0.1, 0.2, 0.3]):
    gabor_feats = []
    for feature in features:
        padded_feature = pad_to_square(feature)
        dim = int(np.sqrt(padded_feature.size))
        image = padded_feature.reshape(dim, dim)
        image_feats = []
        for freq in frequencies:
            _, gabor_resp = gabor(image, frequency=freq)
            image_feats.append(gabor_resp.flatten())
        gabor_feats.append(np.concatenate(image_feats))
    return np.array(gabor_feats)

# Workflow for LDA and Gabor Feature Score Computation
def compute_gabor_scores(test_features_path):
    # Load test features and labels
    test_data = torch.load(test_features_path)

    def to_numpy(data):
        if isinstance(data, torch.Tensor):
            return data.numpy()
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unsupported data type")

    test_feats_np = to_numpy(test_data['embeddings'])
    test_labels_np = to_numpy(test_data['labels'])

    # Compute Gabor features from test embeddings
    gabor_features = compute_gabor_features(test_feats_np)

    # **LDA Score on Gabor Features**
    lda = LDA(shrinkage=0.1)
    lda.fit(gabor_features, test_labels_np)  # Use Gabor features for fitting
    lda_probabilities = lda.predict_proba(gabor_features)

    # Average probability of correct class
    lda_score = np.sum(lda_probabilities[np.arange(len(test_labels_np)), test_labels_np]) / len(test_labels_np)
    print(f"Gabor-LDA Score: {lda_score}")

    # **Energy Score on Gabor Features**
    logits = np.dot(gabor_features, lda.coef_.T) + lda.intercept_
    logits_torch = torch.tensor(logits)

    # Full Energy Score
    full_energy_score = Energy_Score(logits_torch, percent=100, tail='bot')
    print(f"Gabor Energy Score (Full): {1/full_energy_score}")

# Model Paths
# model_paths = {
#     "uni": "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/uni_dhmc_lung.pth",
#     "conch": "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/conch_dhmc_lung.pth",
#     "giga": "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/giga_dhmc_lung.pth",
#     "phikon": "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/phikon_dhmc_lung.pth",
#     "virchow": "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/vir_dhmc_lung.pth",
# }


# model_paths = {
#        "phikon": "/home/jovyan/work/PSE_IMP/imp_pth/giga_imp.pth",
# }

# model_paths = {
#     "uni": "/home/jovyan/work/PSE_CAM/cam_pth/uni_cam.pth",
#     "conch": "/home/jovyan/work/PSE_CAM/cam_pth/conch_cam.pth",
#     "giga": "/home/jovyan/work/PSE_CAM/cam_pth/giga_cam.pth",
#     "phikon": "/home/jovyan/work/PSE_CAM/cam_pth/phikon_cam.pth",
#     "virchow": "/home/jovyan/work/PSE_CAM/cam_pth/vir_cam.pth",
# }

# model_paths = {
#     "uni": "/home/jovyan/work/tran_est/saved_models_and_features_uni_bracs01/uni_vit_large_patch16_pretrained_features.pth",
#     "conch": "/home/jovyan/work/tran_est/saved_models_and_features_conch_bracs01/conch_ViT-B-16_pretrained_features.pth",
#     "giga": "/home/jovyan/work/tran_est/saved_models_and_features_giga_bracs01/giga_model_vit_large_patch16_224_pretrained_features.pth",
#     "phikon": "/home/jovyan/work/tran_est/saved_models_and_features_phikon_bracs00/phikon_v2_train_features.pth",
#     "virchow": "/home/jovyan/work/tran_est/saved_models_and_features_vir_bracs01/Virchow2_pretrained_features.pth",
# }

# model_paths = {
#     "uni": "/home/jovyan/work/tran_est/saved_models_and_features_uni_bach01/uni_vit_large_patch16_pretrained_features.pth",
#     "conch": "/home/jovyan/work/tran_est/saved_models_and_features_conch_bach01/conch_ViT-B-16_pretrained_features.pth",
#     "giga": "/home/jovyan/work/tran_est/saved_models_and_features_giga_bach01/giga_model_vit_large_patch16_224_pretrained_features.pth",
#     "phikon": "/home/jovyan/work/tran_est/saved_models_and_features_phikon_bach01/phikon_v2_train_features.pth",
#     "virchow": "/home/jovyan/work/tran_est/saved_models_and_features_vir_bach01/Virchow2_pretrained_features.pth",
# }

model_paths = {
    "uni": "/home/jovyan/work/tran_est/saved_models_and_features_uni_lc02/uni_vit_large_patch16_pretrained_features.pth",
    "conch": "/home/jovyan/work/tran_est/saved_models_and_features_conch_lc01/conch_ViT-B-16_pretrained_features.pth",
    "giga": "/home/jovyan/work/tran_est/saved_models_and_features_giga_lc02/giga_model_vit_large_patch16_224_pretrained_features.pth",
    "phikon": "/home/jovyan/work/tran_est/saved_models_and_features_phikon_lc02/phikon_v2_train_features.pth",
    "virchow": "/home/jovyan/work/tran_est/saved_models_and_features_vir_lc02/Virchow2_pretrained_features.pth",
}



def compute_scores_for_all_models(model_paths):
    for model_name, path in model_paths.items():
        print(f"\nProcessing model: {model_name}")
        print(f"Feature path: {path}")
        try:
            compute_gabor_scores(path)
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")

# Execute for all models
compute_scores_for_all_models(model_paths)
