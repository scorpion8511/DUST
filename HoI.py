# ### compute histogram metric ##############################

# import numpy as np
# import torch

# # LDA Class (Provided Above)
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

#     def fit(self, X, y):
#         self.classes_ = np.unique(y)
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
#         evals, evecs = np.linalg.eigh(np.linalg.inv(shrunk_Sw).dot(Sb))
#         evecs = evecs[:, np.argsort(evals)[::-1]]
#         self.scalings_ = evecs
#         self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
#         self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(self.priors_)

#     def predict_proba(self, X):
#         logits = np.dot(X, self.coef_.T) + self.intercept_
#         return self.softmax(logits)

#     def softmax(self, X, copy=True):
#         if copy:
#             X = np.copy(X)
#         max_prob = np.max(X, axis=1).reshape((-1, 1))
#         X -= max_prob
#         np.exp(X, X)
#         sum_prob = np.sum(X, axis=1).reshape((-1, 1))
#         X /= sum_prob
#         return X

# # Histogram Intersection Metric Functions
# def compute_histograms(embeddings, labels, num_bins=50):
#     classes = np.unique(labels)
#     class_histograms = {}
#     bin_edges = None

#     for cls in classes:
#         class_embeddings = embeddings[labels == cls]
#         hist, bin_edges = np.histogram(class_embeddings.flatten(), bins=num_bins, density=True)
#         class_histograms[cls] = hist

#     return class_histograms, bin_edges

# def histogram_intersection(hist1, hist2):
#     return np.sum(np.minimum(hist1, hist2))

# def compute_histogram_intersection_metric(embeddings, labels, num_bins=50):
#     class_histograms, _ = compute_histograms(embeddings, labels, num_bins=num_bins)
#     classes = list(class_histograms.keys())
#     num_classes = len(classes)

#     intersection_scores = []
#     for i in range(num_classes):
#         for j in range(i + 1, num_classes):
#             hist1 = class_histograms[classes[i]]
#             hist2 = class_histograms[classes[j]]
#             intersection = histogram_intersection(hist1, hist2)
#             intersection_scores.append(intersection)

#     average_intersection = np.mean(intersection_scores)
#     confidence_score = 1.0 - average_intersection
#     return confidence_score

# # Workflow for All Models
# def compute_scores_for_all_models(model_paths):
#     for model_name, test_features_path in model_paths.items():
#         print(f"Processing model: {model_name}")
#         try:
#             test_data = torch.load(test_features_path)

#             def to_numpy(data):
#                 if isinstance(data, torch.Tensor):
#                     return data.numpy()
#                 elif isinstance(data, np.ndarray):
#                     return data
#                 else:
#                     raise ValueError("Unsupported data type")

#             test_feats_np = to_numpy(test_data['embeddings'])
#             test_labels_np = to_numpy(test_data['labels'])

#             lda = LDA(shrinkage=0.1)
#             lda.fit(test_feats_np, test_labels_np)
#             embeddings = np.dot(test_feats_np, lda.coef_.T) + lda.intercept_

#             confidence_score = compute_histogram_intersection_metric(embeddings, test_labels_np, num_bins=50)
#             print(f"Histogram Intersection Confidence Score for {model_name}: {confidence_score}")

#         except Exception as e:
#             print(f"Error processing model {model_name}: {e}")




# # Example Usage
# # model_paths = {
# #     "uni": "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/uni_dhmc_lung.pth",
# #     "conch": "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/conch_dhmc_lung.pth",
# #     "giga": "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/giga_dhmc_lung.pth",
# #     "phikon": "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/phikon_dhmc_lung.pth",
# #     "virchow": "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/vir_dhmc_lung.pth",
# # }


# # model_paths = {
# #     "uni": "/home/jovyan/work/PSE_CAM/cam_pth/uni_cam.pth",
# #     "conch": "/home/jovyan/work/PSE_CAM/cam_pth/conch_cam.pth",
# #     "giga": "/home/jovyan/work/PSE_CAM/cam_pth/giga_cam.pth",
# #     "phikon": "/home/jovyan/work/PSE_CAM/cam_pth/phikon_cam.pth",
# #     "virchow": "/home/jovyan/work/PSE_CAM/cam_pth/vir_cam.pth",
# # }



# # model_paths = {
# #     "uni": "/home/jovyan/work/tran_est/saved_models_and_features_uni_bracs01/uni_vit_large_patch16_pretrained_features.pth",
# #     "conch": "/home/jovyan/work/tran_est/saved_models_and_features_conch_bracs01/conch_ViT-B-16_pretrained_features.pth",
# #     "giga": "/home/jovyan/work/tran_est/saved_models_and_features_giga_bracs01/giga_model_vit_large_patch16_224_pretrained_features.pth",
# #     "phikon": "/home/jovyan/work/tran_est/saved_models_and_features_phikon_bracs00/phikon_v2_train_features.pth",
# #     "virchow": "/home/jovyan/work/tran_est/saved_models_and_features_vir_bracs01/Virchow2_pretrained_features.pth",
# # }

# # model_paths = {
# #     "uni": "/home/jovyan/work/tran_est/saved_models_and_features_uni_bach01/uni_vit_large_patch16_pretrained_features.pth",
# #     "conch": "/home/jovyan/work/tran_est/saved_models_and_features_conch_bach01/conch_ViT-B-16_pretrained_features.pth",
# #     "giga": "/home/jovyan/work/tran_est/saved_models_and_features_giga_bach01/giga_model_vit_large_patch16_224_pretrained_features.pth",
# #     "phikon": "/home/jovyan/work/tran_est/saved_models_and_features_phikon_bach01/phikon_v2_train_features.pth",
# #     "virchow": "/home/jovyan/work/tran_est/saved_models_and_features_vir_bach01/Virchow2_pretrained_features.pth",
# # }

# model_paths = {
#     "uni": "/home/jovyan/work/tran_est/saved_models_and_features_uni_lc02/uni_vit_large_patch16_pretrained_features.pth",
#     "conch": "/home/jovyan/work/tran_est/saved_models_and_features_conch_lc01/conch_ViT-B-16_pretrained_features.pth",
#     "giga": "/home/jovyan/work/tran_est/saved_models_and_features_giga_lc02/giga_model_vit_large_patch16_224_pretrained_features.pth",
#     "phikon": "/home/jovyan/work/tran_est/saved_models_and_features_phikon_lc02/phikon_v2_train_features.pth",
#     "virchow": "/home/jovyan/work/tran_est/saved_models_and_features_vir_lc02/Virchow2_pretrained_features.pth",
# }


# compute_scores_for_all_models(model_paths)



#######################################################################################################################
#######################################################################################################################

############## histogram image generatopn ############################################################


# import numpy as np
# import torch

# # LDA Class (Provided Above)
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

#     def fit(self, X, y):
#         self.classes_ = np.unique(y)
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
#         evals, evecs = np.linalg.eigh(np.linalg.inv(shrunk_Sw).dot(Sb))
#         evecs = evecs[:, np.argsort(evals)[::-1]]
#         self.scalings_ = evecs
#         self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
#         self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(self.priors_)

#     def predict_proba(self, X):
#         logits = np.dot(X, self.coef_.T) + self.intercept_
#         return self.softmax(logits)

#     def softmax(self, X, copy=True):
#         if copy:
#             X = np.copy(X)
#         max_prob = np.max(X, axis=1).reshape((-1, 1))
#         X -= max_prob
#         np.exp(X, X)
#         sum_prob = np.sum(X, axis=1).reshape((-1, 1))
#         X /= sum_prob
#         return X

# # Histogram Intersection Metric Functions
# def compute_histograms(embeddings, labels, num_bins=50):
#     """
#     Compute class-wise histograms for embeddings.
    
#     Parameters:
#     - embeddings: NumPy array of shape (n_samples, n_features).
#     - labels: NumPy array of shape (n_samples,), corresponding class labels.
#     - num_bins: Number of bins to use for histograms.
    
#     Returns:
#     - class_histograms: Dictionary mapping class labels to histograms (one per class).
#     - bin_edges: Edges of the histogram bins (shared across all classes).
#     """
#     classes = np.unique(labels)
#     class_histograms = {}
#     bin_edges = None

#     for cls in classes:
#         class_embeddings = embeddings[labels == cls]  # Select embeddings for the class
#         hist, bin_edges = np.histogram(class_embeddings.flatten(), bins=num_bins, density=True)
#         class_histograms[cls] = hist  # Store normalized histogram

#     return class_histograms, bin_edges

# def histogram_intersection(hist1, hist2):
#     """
#     Compute the intersection between two histograms.
    
#     Parameters:
#     - hist1: First histogram (array).
#     - hist2: Second histogram (array).
    
#     Returns:
#     - intersection: Intersection value between the two histograms.
#     """
#     return np.sum(np.minimum(hist1, hist2))

# def compute_histogram_intersection_metric(embeddings, labels, num_bins=50):
#     """
#     Compute the histogram intersection metric as a confidence score.
    
#     Parameters:
#     - embeddings: NumPy array of shape (n_samples, n_features).
#     - labels: NumPy array of shape (n_samples,), corresponding class labels.
#     - num_bins: Number of bins to use for histograms.
    
#     Returns:
#     - confidence_score: Overall confidence score based on histogram intersections.
#     """
#     class_histograms, _ = compute_histograms(embeddings, labels, num_bins=num_bins)
#     classes = list(class_histograms.keys())
#     num_classes = len(classes)

#     # Compute pairwise histogram intersections
#     intersection_scores = []
#     for i in range(num_classes):
#         for j in range(i + 1, num_classes):
#             hist1 = class_histograms[classes[i]]
#             hist2 = class_histograms[classes[j]]
#             intersection = histogram_intersection(hist1, hist2)
#             intersection_scores.append(intersection)

#     # Compute overall confidence score (inverse of average intersection)
#     average_intersection = np.mean(intersection_scores)
#     confidence_score = 1.0 - average_intersection  # Higher confidence = lower overlap
#     return confidence_score

# # Example Workflow
# def compute_scores_with_histogram_metric(test_features_path):
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

#     # Fit LDA
#     lda = LDA(shrinkage=0.1)
#     lda.fit(test_feats_np, test_labels_np)
#     embeddings = np.dot(test_feats_np, lda.coef_.T) + lda.intercept_

#     # Compute Histogram Intersection Metric
#     confidence_score = compute_histogram_intersection_metric(embeddings, test_labels_np, num_bins=50)
#     print(f"Histogram Intersection Confidence Score: {confidence_score}")

# # Example Usage
# # test_features_path =  "/home/jovyan/work/tran_est/saved_models_and_features_uni_brk01/uni_vit_large_patch16_pretrained_features.pth"
# # test_features_path =  "/home/jovyan/work/PSE_bncb/dhmc_bncb_pth/vir_bncb.pth"
# # test_features_path =  "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/vir_dhmc_lung.pth"

# uni = "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/uni_dhmc_lung.pth"
# conch= "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/conch_dhmc_lung.pth"
# giga = "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/giga_dhmc_lung.pth"
# phikon= "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/phikon_dhmc_lung.pth"
# virchow = "/home/jovyan/work/PSE_dhmc_lung/dhmc_lung_pth/vir_dhmc_lung.pth"





#############compute_scores_with_histogram_metric(test_features_path) ###############



# import numpy as np
# import matplotlib.pyplot as plt
# import torch

# # LDA Class (Same as Provided Above)
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

#     def fit(self, X, y):
#         self.classes_ = np.unique(y)
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
#         evals, evecs = np.linalg.eigh(np.linalg.inv(shrunk_Sw).dot(Sb))
#         evecs = evecs[:, np.argsort(evals)[::-1]]
#         self.scalings_ = evecs
#         self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
#         self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(self.priors_)

#     def predict_proba(self, X):
#         logits = np.dot(X, self.coef_.T) + self.intercept_
#         return self.softmax(logits)

#     def softmax(self, X, copy=True):
#         if copy:
#             X = np.copy(X)
#         max_prob = np.max(X, axis=1).reshape((-1, 1))
#         X -= max_prob
#         np.exp(X, X)
#         sum_prob = np.sum(X, axis=1).reshape((-1, 1))
#         X /= sum_prob
#         return X

# # Plot and Save Histograms
# # def plot_and_save_histograms(embeddings, labels, num_bins=50, save_path="histograms.png"):
# #     classes = np.unique(labels)
# #     colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

# #     plt.figure(figsize=(10, 6))

# #     for i, cls in enumerate(classes):
# #         class_embeddings = embeddings[labels == cls]
# #         hist, bin_edges = np.histogram(class_embeddings.flatten(), bins=num_bins, density=True)
# #         plt.hist(
# #             bin_edges[:-1], bins=bin_edges, weights=hist, alpha=0.6, label=f"Class {cls}", color=colors[i % len(colors)]
# #         )

# #     plt.xlabel("Feature Value")
# #     plt.ylabel("Density")
# #     plt.title("Class-wise Histograms")
# #     plt.legend()
# #     plt.savefig(save_path)
# #     plt.close()
# #     print(f"Histogram saved to {save_path}")

# # def plot_and_save_histograms(embeddings, labels, num_bins=50, save_path="histograms.png"):
# #     classes = np.unique(labels)
# #     colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

# #     plt.figure(figsize=(12, 8))  # Increased figure size for better visibility

# #     for i, cls in enumerate(classes):
# #         class_embeddings = embeddings[labels == cls]
# #         hist, bin_edges = np.histogram(class_embeddings.flatten(), bins=num_bins, density=True)
        
# #         plt.hist(
# #             bin_edges[:-1], bins=bin_edges, weights=hist, alpha=0.75, 
# #             label=f"Class {cls}", color=colors[i % len(colors)]
# #         )

# #     # Improve axis labels and title visibility
# #     plt.xlabel("Feature Value", fontsize=14, fontweight="bold")
# #     plt.ylabel("Density", fontsize=14, fontweight="bold")
# #     plt.title("Class-wise Histograms", fontsize=16, fontweight="bold")

# #     # Improve axis visibility
# #     plt.xticks(fontsize=12)
# #     plt.yticks(fontsize=12)
# #     plt.grid(True, linestyle="--", alpha=0.5)  # Add a grid for better readability

# #     # Improve legend readability
# #     plt.legend(fontsize=12, frameon=True, loc="upper right")

# #     # Save and show the histogram
# #     plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save with high resolution
# #     plt.close()
# #     print(f"Histogram saved to {save_path}")


# def plot_and_save_histograms(embeddings, labels, num_bins=50, save_path="histograms.png"):
#     classes = np.unique(labels)
#     colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

#     plt.figure(figsize=(14, 8))  # Increased figure size for better readability

#     for i, cls in enumerate(classes):
#         class_embeddings = embeddings[labels == cls]
#         hist, bin_edges = np.histogram(class_embeddings.flatten(), bins=num_bins, density=True)
        
#         plt.hist(
#             bin_edges[:-1], bins=bin_edges, weights=hist, alpha=0.75, 
#             label=f"Class {cls}", color=colors[i % len(colors)]
#         )

#     # Improve axis labels and title visibility
#     plt.xlabel("Feature Value", fontsize=18, fontweight="bold")
#     plt.ylabel("Density", fontsize=18, fontweight="bold")
#     plt.title("Class-wise Histograms", fontsize=20, fontweight="bold")

#     # Improve axis tick readability
#     plt.xticks(fontsize=14, fontweight="bold", rotation=45)  # Rotate x-axis labels for clarity
#     plt.yticks(fontsize=14, fontweight="bold")

#     # Improve legend readability (Bold Font)
#     legend = plt.legend(fontsize=14, frameon=True, loc="upper right")
#     for text in legend.get_texts():
#         text.set_fontweight("bold")  # Set legend text to bold

#     # Adjust limits to ensure proper spacing (optional)
#     plt.ylim(0, None)  # Ensure y-axis starts from 0

#     # Save with very high resolution
#     plt.savefig(save_path, dpi=400, bbox_inches="tight")  # Save in high resolution
#     plt.close()
#     print(f"Histogram saved to {save_path}")



# # Example Workflow
# def compute_and_visualize_histograms(test_features_path, save_path="histograms.png"):
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

#     # Fit LDA
#     lda = LDA(shrinkage=0.1)
#     lda.fit(test_feats_np, test_labels_np)
#     embeddings = np.dot(test_feats_np, lda.coef_.T) + lda.intercept_

#     # Plot and save histograms
#     plot_and_save_histograms(embeddings, test_labels_np, save_path=save_path)

# # Example Usage
# # test_features_path = "/home/jovyan/work/tran_est/saved_models_and_features_giga0010/giga_model_vit_large_patch16_224_pretrained_features.pth"
# test_features_path = "/home/jovyan/work/tran_est/saved_models_and_features_giga_bracs01/giga_model_vit_large_patch16_224_pretrained_features.pth"

# save_path = "bracs_histograms_giga03.png"
# compute_and_visualize_histograms(test_features_path, save_path)




import numpy as np
import matplotlib.pyplot as plt
import torch

# Define model paths
# model_paths = {
#     "giga": "/home/jovyan/work/tran_est/saved_models_and_features_giga_bracs01/giga_model_vit_large_patch16_224_pretrained_features.pth",
#     "conch": "/home/jovyan/work/tran_est/saved_models_and_features_conch_bracs01/conch_ViT-B-16_pretrained_features.pth",
#     "phikon": "/home/jovyan/work/tran_est/saved_models_and_features_phikon_bracs00/phikon_v2_train_features.pth",
#     "uni": "/home/jovyan/work/tran_est/saved_models_and_features_uni_bracs01/uni_vit_large_patch16_pretrained_features.pth",
#     "virchow": "/home/jovyan/work/tran_est/saved_models_and_features_vir_bracs01/Virchow2_pretrained_features.pth",
# }


model_paths = {
    "giga": "/home/jovyan/work/PSE_IMP/imp_pth/giga_imp.pth",
    "conch": "/home/jovyan/work/PSE_IMP/imp_pth/conch_imp.pth",
    "phikon": "/home/jovyan/work/PSE_IMP/imp_pth/phikon_imp.pth",
    "uni": "/home/jovyan/work/PSE_IMP/imp_pth/uni_imp.pth",
    "virchow": "/home/jovyan/work/PSE_IMP/imp_pth/vir_imp.pth",
}


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

    def fit(self, X, y):
        self.classes_ = np.unique(y)
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
        evals, evecs = np.linalg.eigh(np.linalg.inv(shrunk_Sw).dot(Sb))
        evecs = evecs[:, np.argsort(evals)[::-1]]
        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(self.priors_)

    def predict_proba(self, X):
        logits = np.dot(X, self.coef_.T) + self.intercept_
        return self.softmax(logits)

    def softmax(self, X, copy=True):
        if copy:
            X = np.copy(X)
        max_prob = np.max(X, axis=1).reshape((-1, 1))
        X -= max_prob
        np.exp(X, X)
        sum_prob = np.sum(X, axis=1).reshape((-1, 1))
        X /= sum_prob
        return X

# Function to plot and save histograms
def plot_and_save_histograms(embeddings, labels, save_path="histogram.png", num_bins=50):
    classes = np.unique(labels)
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

    global_min = embeddings.min()
    global_max = embeddings.max()
    bin_edges = np.linspace(global_min, global_max, num_bins + 1)

    plt.figure(figsize=(14, 8))  # Increased figure size for better readability

    for i, cls in enumerate(classes):
        class_embeddings = embeddings[labels == cls]
        hist, _ = np.histogram(class_embeddings.flatten(), bins=bin_edges, density=True)

        plt.hist(
            bin_edges[:-1], bins=bin_edges, weights=hist, alpha=0.75,
            label=f"Class {cls}", color=colors[i % len(colors)]
        )

    plt.xlabel("Feature Value", fontsize=18, fontweight="bold")
    plt.ylabel("Density", fontsize=18, fontweight="bold")
    plt.title("Class-wise Histograms", fontsize=20, fontweight="bold")

    plt.xticks(fontsize=14, fontweight="bold", rotation=45)
    plt.yticks(fontsize=14, fontweight="bold")

    legend = plt.legend(fontsize=14, frameon=True, loc="upper right")
    for text in legend.get_texts():
        text.set_fontweight("bold")

    plt.ylim(0, None)
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Histogram saved to {save_path}")


def compute_histogram_intersection_metric(embeddings, labels, num_bins=50):
    global_min = embeddings.min()
    global_max = embeddings.max()
    bin_edges = np.linspace(global_min, global_max, num_bins + 1)

    classes = np.unique(labels)
    histograms = []
    for cls in classes:
        class_embeddings = embeddings[labels == cls]
        hist, _ = np.histogram(class_embeddings.flatten(), bins=bin_edges, density=True)
        histograms.append(hist)

    intersections = []
    for i in range(len(histograms)):
        for j in range(i + 1, len(histograms)):
            intersections.append(np.sum(np.minimum(histograms[i], histograms[j])))

    average_intersection = np.mean(intersections) if intersections else 0.0
    return 1.0 - average_intersection

# Function to process each model and compute histograms
def compute_and_visualize_histograms(test_features_path, save_path, num_bins=50):
    try:
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

        lda = LDA(shrinkage=0.1)
        lda.fit(test_feats_np, test_labels_np)
        embeddings = np.dot(test_feats_np, lda.coef_.T) + lda.intercept_

        plot_and_save_histograms(embeddings, test_labels_np, save_path=save_path, num_bins=num_bins)
        score = compute_histogram_intersection_metric(embeddings, test_labels_np, num_bins=num_bins)
        print(f"Histogram Intersection Score: {score}")

    except Exception as e:
        print(f"Error processing {test_features_path}: {e}")

# Iterate over all models and generate histograms
if __name__ == "__main__":
    for model_name, model_path in model_paths.items():
        save_path = f"histograms_slide_{model_name}.png"
        print(f"Processing model: {model_name}")
        compute_and_visualize_histograms(model_path, save_path)





### UMAP ####################################################

# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import umap

# # LDA Class (Same as Provided Above)
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

#     def fit(self, X, y):
#         self.classes_ = np.unique(y)
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
#         evals, evecs = np.linalg.eigh(np.linalg.inv(shrunk_Sw).dot(Sb))
#         evecs = evecs[:, np.argsort(evals)[::-1]]
#         self.scalings_ = evecs
#         self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
#         self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(self.priors_)

# # Function to plot and save UMAP

# def plot_and_save_umap(embeddings, labels, save_path="umap.png"):
#     reducer = umap.UMAP(n_components=2, random_state=42)
#     umap_embeds = reducer.fit_transform(embeddings)
    
#     plt.figure(figsize=(10, 8))
#     classes = np.unique(labels)
#     colors = plt.cm.tab10.colors
    
#     for i, cls in enumerate(classes):
#         plt.scatter(
#             umap_embeds[labels == cls, 0], 
#             umap_embeds[labels == cls, 1],
#             label=f"Class {cls}",
#             color=colors[i % len(colors)],
#             alpha=0.7
#         )
    
#     plt.xlabel("UMAP Component 1", fontsize=14, fontweight="bold")
#     plt.ylabel("UMAP Component 2", fontsize=14, fontweight="bold")
#     plt.title("UMAP Projection of Feature Embeddings", fontsize=16, fontweight="bold")
#     plt.legend(fontsize=12)
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"UMAP visualization saved to {save_path}")

# # Example Workflow
# def compute_and_visualize_features(test_features_path, hist_save_path="histograms.png", umap_save_path="umap.png"):
#     test_data = torch.load(test_features_path)
    
#     def to_numpy(data):
#         if isinstance(data, torch.Tensor):
#             return data.numpy()
#         elif isinstance(data, np.ndarray):
#             return data
#         else:
#             raise ValueError("Unsupported data type")
    
#     test_feats_np = to_numpy(test_data['embeddings'])
#     test_labels_np = to_numpy(test_data['labels'])
    
#     lda = LDA(shrinkage=0.1)
#     lda.fit(test_feats_np, test_labels_np)
#     embeddings = np.dot(test_feats_np, lda.coef_.T) + lda.intercept_
    
#     # Plot and save UMAP visualization
#     plot_and_save_umap(embeddings, test_labels_np, save_path=umap_save_path)

# # Example Usage
# # test_features_path = "/home/jovyan/work/tran_est/saved_models_and_features_giga_bracs01/giga_model_vit_large_patch16_224_pretrained_features.pth"
# # umap_save_path = "bracs_umap_giga03.png"

# test_features_path = "/home/jovyan/work/tran_est/saved_models_and_features_phikon_lc02/phikon_v2_train_features.pth"
# umap_save_path = "lc_umap_phikon.png"

# compute_and_visualize_features(test_features_path, umap_save_path=umap_save_path)


