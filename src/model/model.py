import os
from copy import deepcopy
from typing import Iterator, List, Tuple, Callable
import hashlib
import time
import numpy as np
import torch
from captum.attr import Saliency, Occlusion
from captum.attr import visualization as viz, DeepLift, IntegratedGradients, NoiseTunnel
from torch.nn import Parameter
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from transformers.models.clip.modeling_clip import CLIPOutput

from src.arguments.env_args import EnvArgs
from src.arguments.model_args import ModelArgs
from src.arguments.outdir_args import OutdirArgs
from src.dataset.dataset import Dataset
from src.utils.python_helper import hash_dict
from src.utils.sklearn_helper import pcp
from src.utils.smooth_value import SmoothedValue
from src.utils.special_images import plot_images
from src.utils.special_print import print_highlighted, print_warning
from src.utils.torch_helper import to_groups
from src.utils.web import is_valid_url


class Model(torch.nn.Module):
    CONFIG_MODEL_ARGS = "model_args_config"
    CONFIG_BASE_MODEL_STATE_DICT = "base_model_state_dict"

    # output modes during the forward(*args) function
    FEATURES = "features"  # return the last feature layer
    FINAL = "final"  # return the logits

    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):
        super().__init__()
        self.model_args: ModelArgs = model_args
        self.env_args: EnvArgs = env_args

        self.model: torch.nn.Module = model_args.get_base_model().to(self.env_args.device).eval()

        # Flags to indicate which inference-time defense to use
        self._use_saliency_defense = False
        self._use_randomized_smoothing = False

        # Flags to control model behavior
        self._output_mode = self.FINAL  # FINAL: Return prediction, FEATURES: return logits
        self._tick = 0
        self._debug_mode = False

        # Data structures that store states of the model
        self._hooks = []
        self._preprocessors = []
        self._feature_recording_hooks = []
        self._custom_hooks = []
        self._hidden_state = {}
        self.print_flags = {}
        self.dim_reduction = {}

        # Tensors
        self._feature_weights: torch.Tensor | None = None
        self._feature_importance: torch.Tensor | None = None

        # Initialization functions
        self.activate_feature_recording()
        if model_args.model_ckpt is not None:
            self.load()

    def get_layer(self, name) -> torch.nn.Module:
        try:
            self.model.__getattr__(name)
        except:
            for layer_name, layer in self.named_modules():
                if layer_name == name:
                    return layer

    def get_feature_layer_names(self) -> List[str]:
        """ Returns the names of all feature layers
        """
        if self.first_time("print_layer_names") and self.model_args.show_layer_names:
            for name, module in self.model.named_modules():
                print(name, module)

        if self.model_args.model_name == "resnet18" and self.model_args.resolution == 32:
            return ["layer1", "layer2", "layer3", "layer4", "linear"]
        elif self.model_args.model_name == "resnet18" and self.model_args.resolution == 224:
            return ["layer1", "layer2", "layer3", "layer4", "fc"]
        elif self.model_args.model_name == "openai/clip-vit-base-patch32" or self.model_args.model_name == "openai/clip-vit-base-patch16"  :
            return ['model']
        elif self.model_args.model_name == "google/vit-base-patch16-224":
            return ['model.classifier']
        else:
            return ["fc"]  # best guess

    def deepcopy(self) -> 'Model':
        """ Creates a copy of this model
        """
        args = self.model_args
        self.model_args = None
        self._disable_feature_recording()
        hidden_state = self._hidden_state
        self._hidden_state = None
        copy = deepcopy(self)
        self._hidden_state = hidden_state
        copy._hidden_state = hidden_state
        self.model_args = args
        copy.model_args = deepcopy(args)
        self.activate_feature_recording()
        copy.activate_feature_recording()
        return copy

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.model.parameters()

    def train(self, mode: bool = True):
        self.model = self.model.train()
        return self

    def get_embeddings(self, dataset: Dataset, centroids=False, verbose: bool = False) -> dict:
        """ Dict: Classes -> Embeddings as torch tensor"""
        self.eval()
        with torch.no_grad():

            data_loader = DataLoader(dataset, batch_size=self.env_args.batch_size,
                                     shuffle=False, num_workers=self.env_args.num_workers, drop_last=True)
            pbar = tqdm(data_loader, disable=not verbose)

            embeddings = {}
            for x, y in pbar:
                self.forward(x.to(self.env_args.device))
                features = self.get_features()
                for y_i, feature in zip(y, features):
                    embeddings[y_i.item()] = embeddings.setdefault(y_i.item(), []) + [feature.cpu().detach()]
            embeddings = {c: torch.stack(x, 0) for c, x in embeddings.items()}

            if centroids:
                from sklearn.cluster import KMeans
                # Fit a KMeans clustering model
                model = KMeans(n_clusters=dataset.num_classes(), random_state=42)

                # turn embeddings from dict to list
                X = []
                for c, x in embeddings.items():
                    X.append(x)
                X = torch.cat(X, 0)
                labels = model.fit_predict(X)

                # Compute the centroid of each cluster
                centroids = model.cluster_centers_

                # Compute the distance between each data_cleaning point and its cluster centroid
                distances = np.zeros(X.shape[0])
                for i in range(X.shape[0]):
                    cluster_idx = labels[i]
                    centroid = centroids[cluster_idx]
                    distances[i] = np.linalg.norm(X[i] - centroid)

                embeddings = {c: torch.from_numpy(centroids[c]) for c in range(dataset.num_classes())}
                # embeddings = {c: torch.mean(x, 0) for c, x in embeddings.items()}
        return embeddings

    def evaluate_class_scores(self, dataset: Dataset) -> float:
        data_loader = DataLoader(dataset, batch_size=self.env_args.batch_size,
                                 shuffle=True, num_workers=self.env_args.num_workers)
        self.eval()
        y_all = torch.zeros(dataset.num_classes()).to(self.env_args.device)
        ctr = 0
        for x, y in data_loader:
            x, y = x.to(self.env_args.device), y.to(self.env_args.device)
            y_pred = self.forward(x)
            y_all += y_pred.sum(0)
            ctr += x.shape[0]
        return y_all / ctr

    def evaluate(self, dataset: Dataset, verbose: bool = False, top_5=False) -> float:
        data_loader = DataLoader(dataset, batch_size=self.env_args.batch_size,
                                 shuffle=True, num_workers=self.env_args.num_validation_workers)
        acc = SmoothedValue()
        self.eval()
        with torch.no_grad():
            pbar = tqdm(data_loader, disable=not verbose)
            for x, y in pbar:
                x, y = x.to(self.env_args.device), y.to(self.env_args.device)
                y_pred = self.forward(x)
                if not top_5:
                    acc.update(self.accuracy(y_pred, y))
                else:
                    acc.update(self.top5_accuracy(y_pred, y))

                pbar.set_description(f"'test_acc': {100 * acc.global_avg:.2f}")
        return acc.global_avg.item()

    def evaluate_with_loss(self, dataset: Dataset, loss_fxn=torch.nn.CrossEntropyLoss(), verbose: bool = False) -> (float, float):
        data_loader = DataLoader(dataset, batch_size=self.env_args.batch_size,
                                 shuffle=True, num_workers=self.env_args.num_validation_workers)
        acc = SmoothedValue()
        self.eval()
        with torch.no_grad():
            av_loss = 0
            pbar = tqdm(data_loader, disable=not verbose)
            for x, y in pbar:
                x, y = x.to(self.env_args.device), y.to(self.env_args.device)
                y_pred = self.forward(x)

                acc.update(self.accuracy(y_pred, y))
                loss = loss_fxn(y_pred, y)
                av_loss = av_loss + loss
                pbar.set_description(f"'test_acc': {100 * acc.global_avg:.2f}")
        return acc.global_avg.item(), (av_loss / len(data_loader)).item()

    def eval(self):
        self.model = self.model.eval()
        return self

    def add_preprocessor(self, preprocessor: Callable):
        self._preprocessors += [preprocessor]

    def activate_feature_recording(self) -> None:
        """ Activates feature recording. """
        feature_layer_names: List[str] = self.get_feature_layer_names()
        self._disable_feature_recording()

        def get_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, CLIPOutput):
                    self._hidden_state[name] = output.image_embeds.clone()
                else:
                    self._hidden_state[name] = input[0].clone()
                return output

            return hook_fn

        for name, layer in self.model.named_modules():
            if name in feature_layer_names:
                self._feature_recording_hooks += [layer.register_forward_hook(get_hook(name))]

    def add_features_hook(self, func, layer_name=None):
        feature_layer_name: str = self.get_feature_layer_names()[-1] if layer_name is None else layer_name

        def feature_hook_fn(module, input, output):
            mod2 = deepcopy(module)
            mod2._forward_hooks.clear()

            if isinstance(output, CLIPOutput):
                output.image_embeds = module(func(output.image_embeds))
            else:
                output = mod2(func(input[0]))
            return output

        def get_hook():
            return feature_hook_fn

        for name, layer in self.model.named_modules():
            if name == feature_layer_name:
                self._custom_hooks += [layer.register_forward_hook(get_hook())]

    def remove_features_hook(self):
        for hook in self._custom_hooks:
            hook.remove()
        self._custom_hooks = []

    def _disable_feature_recording(self) -> None:
        for hook in self._feature_recording_hooks:
            hook.remove()
        self._feature_recording_hooks = []

    def get_features(self, name: None | List | str = None) -> torch.Tensor | List:
        """ Returns the features on the last inference of the model.
        """
        try:
            if name is None:  # return last layer
                return self._hidden_state[self.get_feature_layer_names()[-1]]
            elif isinstance(name, list):  # return all layers in list
                return [self._hidden_state[n_i] for n_i in name]
            else:  # return one layer
                return self._hidden_state[name]
        except Exception as e:
            print(
                f"Could not find layer: {e}. Available layers: {[name for name, _ in self.model.named_modules() if len(name) > 0]}")

            exit()

    def _restoration(self, x: torch.Tensor, saliency_map: torch.Tensor):
        """ Restores an image given a saliency map.
        """
        x_prime = transforms.RandomCrop(x.shape[-1], padding=int((4 / 32) * x.shape[-1]))(x)
        if self.model_args.saliency_defense_mode == "threshold":
            raise NotImplementedError
        elif self.model_args.saliency_defense_mode == "topk":
            original_size = saliency_map.shape
            _, index = torch.topk(saliency_map.flatten(2), self.model_args.saliency_topk)

            # Compute boolean map of top-k values
            flattened_x, flattened_x_prime = x.flatten(2), x_prime.flatten(2)
            bool_index = torch.zeros_like(flattened_x, dtype=torch.bool)
            for j in range(original_size[0]):
                indices = index[j, 0]
                bool_index[j, :, indices] = True
            flattened_x = self._pixel_inpainting(flattened_x, bool_index, flattened_x_prime)
            x = flattened_x.unflatten(2, (original_size[-2], original_size[-1]))
        else:
            raise NotImplementedError
        return x

    def _pixel_inpainting(self, x: torch.Tensor, indices: torch.BoolTensor, x_prime: torch.Tensor):
        """ Perform pixel removal according to the pixel removal method selected in the model args.
        x (and x_prime for neighbour removal) needs to be able to be indexed by indices.
        """
        if self.model_args.saliency_removal_method == "neighbour":
            # replace with nearby pixels
            x[indices] = x_prime[indices]
        elif self.model_args.saliency_removal_method == "erasure":
            # erase pixels
            x[indices] = 0
        elif self.model_args.saliency_removal_method == "random":
            # replace with random values
            rand = torch.rand(size=x[indices].shape).to(x.device)
            # project [0, 1) to [a, b) where a is the min value in x_dash and b is the max_value in x_dash
            min_value = torch.min(x)
            max_value = torch.max(x)
            rand = rand * (max_value - min_value) + min_value
            x[indices] = rand
        return x

    def forward_randomized_smoothing(self, x: torch.tensor):
        """ Randomized smoothing at inference time.
        """
        self.activate_randomized_smoothing(False)
        y_pred = 0
        for _ in range(self.model_args.smoothing_reps):
            x = deepcopy(x)
            noise = torch.normal(mean=0, std=self.model_args.smoothing_sigma,
                                 size=x.shape, device=self.env_args.device)
            x = x.add(noise)
            y_pred = torch.softmax(self(x), -1) + y_pred

        self.activate_randomized_smoothing(True)
        return y_pred

    def get_feature_importance(self, y_pred: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        if self._feature_importance is None:
            weight_mat = None
            for param in self.model.parameters():
                if len(param.shape) == 2:  # hacky way of finding the last layer's weights
                    weight_mat = param.data.detach()

            if weight_mat is None:
                print_warning(f"Could not automatically find the feature weights for this model! "
                              f"Resorting to non-importance sampling. ")
                self._feature_importance = torch.ones(target_shape)
            else:
                self._feature_importance = torch.stack([weight_mat[i.item()].cpu() for i in y_pred.argmax(1)], 0)
        return deepcopy(self._feature_importance)

    def forward_saliency_defence(self, x: torch.tensor):
        """ Activates our inference-time defense.
        """
        self.activate_saliency_defense(False)  # To avoid infinite recursion
        original_mode = self.get_output_mode()
        self.set_output_mode(self.FINAL)
        y_pred0 = self.model(x)
        features = self._hidden_state[self.get_feature_layer_names()[-1]]
        self.set_output_mode(self.FEATURES)

        if self.model_args.heatmap_algorithm == "noise_tunnel":
            if self.first_time("warn_about_noise_tunnel"):
                print_warning(f"Noise Tunnel may crash due to high memory requirements .. ")

        y_pred = 0
        if self.model_args.saliency_defense_method == "weighted":
            repetitions = self.model_args.saliency_repetitions
            n = self.model_args.saliency_num_neurons

            for rep_ctr in range(repetitions):
                num_features = features.shape[-1]
                self._feature_weights = self.get_feature_importance(y_pred0, (len(x), num_features))

                # -- Randomly sample features and set the feature weight tensor
                mask = torch.zeros_like(self._feature_weights)
                for j in range(len(x)):
                    idx = np.random.choice(np.arange(num_features), size=n)
                    mask[j, idx] = 1
                self._feature_weights *= mask

                # -- Compute normed saliency map (always use target class zero to group features)
                self.set_output_mode(self.SALIENCY_DEFENSE)
                saliency_map = self.get_saliency_map(deepcopy(x), [0] * len(x))
                '''for j in range(len(saliency_map)):
                    norm = torch.linalg.norm(saliency_map[j])
                    saliency_map[j] = saliency_map[j] / norm
                    if torch.isnan(saliency_map[j].mean()):
                        saliency_map[j] = torch.zeros_like(saliency_map[j])'''
                saliency_map = saliency_map.sum(1).unsqueeze(1).repeat_interleave(3, 1)

                # -- Input restoration. x_r is the restored input.
                x_r = deepcopy(x)

                x_r = self._restoration(x_r, saliency_map)
                self.set_output_mode(self.FINAL)
                new_pred = torch.softmax(self.forward(x_r), 1)
                y_pred = new_pred + y_pred

                if rep_ctr == 0 and self._debug_mode:
                    if self.first_time(f"{self._tick}_show_restoration"):
                        plot_images(torch.cat([x[:3], x_r[:3], 10 * (x[:3] - x_r[:3])], 0), n_row=3)
                    # print(y_pred.argmax(1))

        else:
            raise ValueError(self.model_args.saliency_defense_method)

        self.activate_saliency_defense(True)
        self.set_output_mode(original_mode)
        return y_pred

    def set_output_mode(self, state: str):
        """ Controls the output of the forward prediction
        """
        assert state in [self.FEATURES, self.FINAL, self.SALIENCY_DEFENSE], "invalid state"
        self._output_mode = state

    def get_output_mode(self):
        return self._output_mode

    def forward(self, x):
        for preprocessor in self._preprocessors:
            x = preprocessor(x)

        if self._use_randomized_smoothing and self._use_saliency_defense:
            if self.first_time("warn_simultaneous_defenses"):
                print_warning(f"More than one defenses are active. They will override each other.")

        if self._use_randomized_smoothing:
            return self.forward_randomized_smoothing(x)
        if self._use_saliency_defense:
            return self.forward_saliency_defence(x)

        if self._output_mode == self.FINAL:
            return self.model(x)
        elif self._output_mode == self.FEATURES:
            self.model(x)
            return self._hidden_state[self.FEATURES]
        elif self._output_mode == self.SALIENCY_DEFENSE:
            y_pred = self.model(x)
            features = self._hidden_state[self.FEATURES]
            try:
                out = (self._feature_weights.to(self.env_args.device) * features).mean(-1).unsqueeze(-1)
            except:
                if self.first_time("return_full_prediction_vector"):
                    print_warning(f"Forward saliency defence returns the entire prediction "
                                  f"vector, but does not perform feature sampling. "
                                  f"This is likely not what we want for certification. Switch"
                                  f"to 'heatmap_algorithm=Saliency' to get rid of this warning")
                out = y_pred  # ToDo: Fix
            return out
        else:
            raise ValueError(self._output_mode)

    @staticmethod
    def accuracy(y_pred, y) -> float:
        return (y_pred.argmax(1) == y).float().mean()

    @staticmethod
    def top5_accuracy(y_pred, y) -> float:
        top5_pred = y_pred.topk(5, 1)[1]
        correct = top5_pred.eq(y.view(-1, 1).expand_as(top5_pred)).sum()
        return correct.float() / y.size(0)

    def save(self, outdir_args: OutdirArgs = None, fn=None) -> dict:
        data = {
            self.CONFIG_MODEL_ARGS: self.model_args,
            self.CONFIG_BASE_MODEL_STATE_DICT: self.model.state_dict()
        }
        if fn is not None:
            torch.save(data, fn)
            print_highlighted(f"Saved model at {os.path.abspath(fn)}")
        elif outdir_args is not None:
            folder = outdir_args._get_folder_path()
            fn = os.path.join(folder, "model.pt")
            print(fn)
            torch.save(data, fn)
            print_highlighted(f"Saved model at {os.path.abspath(fn)}")
        return data

    def activate_saliency_defense(self, state: bool = True) -> None:
        """ Activates the saliency defense in this model.
        """
        self._use_saliency_defense = state

    def activate_randomized_smoothing(self, state: bool = True) -> None:
        """ Activates randomized smoothing
        """
        self._use_randomized_smoothing = state

    def debug(self, mode: bool = True) -> None:
        self._debug_mode = mode

    def debug_tick(self) -> None:
        """ Clears plotting for debugs. """
        self._tick += 1

    def first_time(self, name) -> bool:
        """ Checks if something has been invoked for the first time """
        state = name not in self.print_flags
        self.print_flags[name] = True
        return state

    def plot_confusion_matrix(self, dataset, normalize=True, title=None):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        cmap = plt.cm.Blues
        self.eval()

        data_loader = DataLoader(dataset, batch_size=self.env_args.batch_size,
                                 drop_last=True, shuffle=False, num_workers=self.env_args.num_workers)
        y_true, y_pred = [], []
        for x, y in data_loader:
            y_true += [y]
            y_pred += [self.forward(x.to(self.env_args.device)).detach()]
        y_pred = torch.cat(y_pred, 0).argmax(1)
        y_true = torch.cat(y_true, 0).long()

        cm = confusion_matrix(y_true.cpu(), y_pred.cpu())
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap=cmap)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j] * 100:.1f}", ha='center', va='center', color='black')
        for j in range(cm.shape[1]):
            ax.text(j, -1, f"{cm[:, j].sum() * 100:.1f}", ha='center', va='center', color='black')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(title)
        fig.colorbar(im)
        plt.show()

    def visualize_latent(self, dataset: Dataset, method='tsne', decision_boundary: bool = False, title=None,
                         savefig: str = None, verbose: bool = True, show: bool = True, preserve_reduction=False):
        """ Visualizes the feature space of this model as a tsne plot """
        assert method in ['tsne', 'mds', 'umap', 'lda', 'pcp', 'pca', 'nmf'], f"Specified method {method} not found .. "
        data_loader = DataLoader(dataset, batch_size=self.env_args.batch_size,
                                 shuffle=True, num_workers=self.env_args.num_workers)

        embeddings: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []

        # 1. Collect features
        self.eval()
        pbar = tqdm(data_loader, disable=not verbose)
        for x, y in pbar:
            x = x.to(self.env_args.device)
            y_pred = self.forward(x)
            embeddings += [self.get_features(self.get_feature_layer_names()[-1]).detach().cpu()]
            # embeddings += [y_pred.detach().cpu()]
            labels += [y.cpu()]
            pbar.set_description(f"Recording features")
        embeddings: torch.Tensor = torch.cat(embeddings, 0)
        labels_before: torch.Tensor = torch.cat(labels, 0)
        labels = to_groups(labels_before)
        target_cls = labels[labels_before == 0][0].item()
        backdoor_cls = labels[labels_before == 1001][0].item()

        # 2. Choose method
        if method == 'tsne':
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            tsne = TSNE(n_components=2, random_state=1)
            x_reduced = tsne.fit_transform(embeddings.squeeze().numpy())
            plt.title("t-distributed Stochastic Neighbor Embedding" if title is None else title)
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
        elif method == 'umap':
            import umap
            import matplotlib.pyplot as plt

            umap_obj = umap.UMAP(n_components=2, random_state=1)
            x_reduced = umap_obj.fit_transform(embeddings.squeeze().numpy())

            plt.title("Uniform Manifold Approximation and Projection" if title is None else title)
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")

        elif method == "mds":
            import matplotlib.pyplot as plt
            from sklearn.manifold import MDS
            mds = MDS(n_components=2, metric=True, random_state=2)
            x_reduced = mds.fit_transform(embeddings.squeeze().numpy())
            plt.title("MDS" if title is None else title)
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
        elif method == "lda":
            import matplotlib.pyplot as plt
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            lda = LinearDiscriminantAnalysis(n_components=2)
            lda_labels = deepcopy(labels)
            lda_labels[lda_labels == backdoor_cls] = target_cls
            lda.fit(embeddings, lda_labels)
            x_reduced = lda.transform(embeddings)

            plt.title("Linear Discriminant Analysis" if title is None else title)
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
        elif method == "pcp":
            import matplotlib.pyplot as plt
            x_reduced = pcp(embeddings)
            plt.title("Principal Component Pursuit" if title is None else title)
            plt.xlabel("PC 1")
            plt.ylabel("PC 2")
        elif method == "pca":
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            if preserve_reduction and self.dim_reduction.setdefault("pca", None) is not None:
                pca = self.dim_reduction["pca"]
                x_reduced = pca.transform(embeddings)
            else:
                pca = PCA(n_components=2)
                x_reduced = pca.fit_transform(embeddings)
                self.dim_reduction["pca"] = pca
            plt.title("Principal Component Analysis" if title is None else title)
            plt.xlabel("PC 1")
            plt.ylabel("PC 2")
        elif method == "nmf":
            from sklearn.decomposition import NMF
            import matplotlib.pyplot as plt
            nmf = NMF(n_components=2, init='nndsvda', random_state=0)
            x_reduced = nmf.fit_transform(embeddings)
            plt.title("NMF" if title is None else title)
            plt.xlabel("PC 1")
            plt.ylabel("PC 2")
        else:
            ValueError(method)

        if decision_boundary:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(solver='lbfgs')
            model.fit(x_reduced, labels)

            # Plot the decision boundaries of the logistic regression model
            x_min, x_max = x_reduced[:, 0].min() - 0.5, x_reduced[:, 0].max() + 0.5
            y_min, y_max = x_reduced[:, 1].min() - 0.5, x_reduced[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                 np.arange(y_min, y_max, 0.1))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.4)

        centroids = {i: torch.mean(torch.from_numpy(x_reduced[labels == i]), dim=0) for i in
                     [idx for idx in range(dataset.num_classes()) if idx in labels]}

        # Fill with data_cleaning.
        labels2 = labels[labels != backdoor_cls]
        non_poison = (
            x_reduced[:, 0][labels != backdoor_cls][labels2 != 0],
            x_reduced[:, 1][labels != backdoor_cls][labels2 != 0])
        poison = (x_reduced[:, 0][labels == backdoor_cls], x_reduced[:, 1][labels == backdoor_cls])
        target = (x_reduced[:, 0][labels == 0], x_reduced[:, 1][labels == 0])
        # make sure that every class always gets the same color
        plt.scatter(*non_poison, c=labels2[labels2 != 0], s=10, alpha=1)

        # add a text for each centroid
        for i in centroids:
            plt.text(centroids[i][0], centroids[i][1], dataset.classes[i].capitalize(), fontsize=12, color='black')

        plt.scatter(*target, marker='s', color='orange', s=10)
        plt.scatter(*poison, marker='x', color='red', s=10)
        if savefig is not None:
            print(f"Saved figure at '{os.path.abspath(savefig)}")
            plt.savefig(savefig)

        if show:
            plt.show()
        else:
            plt.clf()

        return None

    def get_saliency_map(self, x: torch.tensor, y: List[int]):
        """ Computes the saliency map using captum on an input with [chw]
        """
        x = deepcopy(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # y = self(torch.tensor(x).to(self.env_args.device)).argmax(dim=1)

        print(x.min(), x.max())

        x.requires_grad = True
        self.model.eval()

        self.model_args.heatmap_algorithm = "integrated_gradients"
        if self.first_time("warn_of_overwriting_heatmap"):
            print_warning(f"Currently overwriting heatmap algorithm with {self.model_args.heatmap_algorithm}")

        if self.model_args.heatmap_algorithm == "saliency":
            saliency = Saliency(self)
            saliency_map = saliency.attribute(x.to(self.env_args.device), target=y)
        elif self.model_args.heatmap_algorithm == "deeplift":
            if self.first_time("warn_about_deeplift"):
                print_warning(f"Deeplift will only work if there are no repeated modules in the model."
                              f"Basically, if this doesn't work you have to modify the base model. ")
            dl = DeepLift(self)
            saliency_map = dl.attribute(x.to(self.env_args.device), target=y,
                                        baselines=torch.zeros_like(x).to(self.env_args.device))

        elif self.model_args.heatmap_algorithm == "integrated_gradients":
            ig = IntegratedGradients(self)
            saliency_map = ig.attribute(x.to(self.env_args.device), target=y,
                                        baselines=torch.zeros_like(x).to(self.env_args.device))
        elif self.model_args.heatmap_algorithm == "noise_tunnel":
            ig = IntegratedGradients(self)
            nt = NoiseTunnel(ig)
            saliency_map = nt.attribute(x.to(self.env_args.device), target=y,
                                        baselines=torch.zeros_like(x).to(self.env_args.device),
                                        nt_type='smoothgrad_sq', nt_samples=10, stdevs=0.2)
        elif self.model_args.heatmap_algorithm == "occlusion":
            occlusion = Occlusion(self)
            saliency_map = occlusion.attribute(x.to(self.env_args.device),
                                               target=y,
                                               strides=(3, 10, 10),
                                               sliding_window_shapes=(3, 20, 20),
                                               baselines=0)
        else:
            raise ValueError(self.model_args.heatmap_algorithm)

        if self._debug_mode:
            if self.first_time(f"{self._tick}_show_gradient"):
                grads = np.transpose(saliency_map[0].squeeze().cpu().detach().numpy(), (1, 2, 0))
                x_plot = np.transpose(x[0].squeeze().cpu().detach().numpy(), (1, 2, 0))
                _ = viz.visualize_image_attr(grads, x_plot, method="blended_heat_map",
                                             alpha_overlay=0.5, show_colorbar=True,
                                             title="Overlayed Gradient Magnitudes (Saliency Map)")
        return saliency_map

    def load(self, content=None, ckpt=None):
        if content is None:
            ckpt = ckpt if ckpt is not None else self.model_args.model_ckpt

            # first, check if it's a valid filepath
            if os.path.exists(ckpt):
                content = torch.load(ckpt, map_location='cpu')
            elif is_valid_url(ckpt):
                content = torch.hub.load_state_dict_from_url(ckpt, progress=False)
            else:
                raise FileNotFoundError
        if ModelArgs.CONFIG_KEY in content.keys():
            content = content[ModelArgs.CONFIG_KEY]

        # Hacky part. See if this is a checkpoint to load the base model or to load this model.
        if self.CONFIG_MODEL_ARGS in content.keys():
            self.model_args = content[self.CONFIG_MODEL_ARGS]
            self.model.load_state_dict(content[self.CONFIG_BASE_MODEL_STATE_DICT])
            self.model.eval()
        else:
            # we assume this is just a state dict for the base model
            self.model.load_state_dict(content)
            self.model.eval()
        self.model_args.model_hash = hash_dict({0: self.state_dict()[list(self.state_dict().keys())[0]]})
        return self



