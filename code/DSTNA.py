import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DSTNA(nn.Module):
  

    def __init__(self, args):
        super(DSTNA, self).__init__()
        self.config = args

        self.init_layers()

        # Memory pools
        self.local_memory_pool = []
        self.global_memory_pool = []

        # Memory configuration
        self.similarity_threshold = getattr(args, 'similarity_threshold', 0.85)
        self.max_local_pool_size = getattr(args, 'local_pool_size', 50)
        self.max_global_pool_size = getattr(args, 'global_pool_size', 50)

        # Drift detection
        self.window_samples = []
        self.window_size = getattr(args, 'window_size', 100)
        self.js_mean = 0.0
        self.js_drift = False

        # Error tracking
        self.error_mean = 0.0
        self.error_std = 1.0

        self.memory_train_freq = getattr(args, 'memory_train_freq', 50)

    def init_layers(self):
        """Initialize network layers and weight vectors."""
        hl_num = self.config.ln - 1

        # Hidden layers
        hidden_layers = [nn.Linear(self.config.nIn, self.config.nHn)]
        hidden_layers.extend([nn.Linear(self.config.nHn, self.config.nHn) for _ in range(hl_num)])
        self.hidden_layers = nn.ModuleList(hidden_layers)

        # Output layers
        output_layers = [nn.Linear(self.config.nHn, self.config.nOut) for _ in range(hl_num)]
        self.output_layers = nn.ModuleList(output_layers)

        # Weight vectors
        self.alpha_vector = torch.full(size=(hl_num,), fill_value=1. / hl_num, requires_grad=False)
        self.gamma_vector = torch.full(size=(hl_num,), fill_value=1., requires_grad=False)
        self.v = torch.full(size=(hl_num,), fill_value=1., requires_grad=False)

    def calculate_correlation(self, x1, x2):
        """Calculate correlation coefficient between two tensors."""
        try:
            x1_flat = x1.flatten().detach().cpu().numpy()
            x2_flat = x2.flatten().detach().cpu().numpy()

            if np.std(x1_flat) == 0 or np.std(x2_flat) == 0:
                return 0

            corr = np.corrcoef(x1_flat, x2_flat)[0, 1]
            return corr if not np.isnan(corr) else 0
        except Exception:
            return 0

    def calculate_max_correlation(self, x):
        """Find maximum correlation with samples in local memory pool."""
        if len(self.local_memory_pool) == 0:
            return -1
        try:
            correlations = [self.calculate_correlation(x, sample[0]) for sample in self.local_memory_pool]
            return max(correlations)
        except Exception:
            return -1

    def update_local_memory(self, x, y):
        """Update local memory pool with new sample."""
        try:
            if self.concept_drift_detect():
                self.local_memory_pool = []

            max_correlation = self.calculate_max_correlation(x)

            if max_correlation < self.similarity_threshold:
                if len(self.local_memory_pool) >= self.max_local_pool_size:
                    self.local_memory_pool.pop(0)
                self.local_memory_pool.append([x, y, 1])
            else:
                correlations = [self.calculate_correlation(x, sample[0]) for sample in self.local_memory_pool]
                max_idx = np.argmax(correlations)
                self.local_memory_pool[max_idx][2] += 1
        except Exception as e:
            print(f"Error updating local memory: {e}")

    def update_global_memory(self, x, y, error):
        """Update global memory pool based on prediction error."""
        try:
            if len(self.global_memory_pool) == 0:
                self.error_mean = error
                self.error_std = error * 0.1
            else:
                self.error_mean = 0.95 * self.error_mean + 0.05 * error
                self.error_std = 0.95 * self.error_std + 0.05 * abs(error - self.error_mean)

            if error > self.error_mean + self.error_std:
                if len(self.global_memory_pool) >= self.max_global_pool_size:
                    self.global_memory_pool.pop(0)
                self.global_memory_pool.append([x, y])
        except Exception as e:
            print(f"Error updating global memory: {e}")

    def calculate_js_divergence(self, preds, labels):
        """Calculate Jensen-Shannon divergence between predictions and labels."""
        if len(preds) < 2:
            return 0

        try:
            pred_classes_list = []
            for pred in preds:
                if isinstance(pred, torch.Tensor):
                    if pred.dim() > 1:
                        if hasattr(self, 'alpha_vector'):
                            weighted_pred = torch.sum(self.alpha_vector.unsqueeze(-1).unsqueeze(-1) * pred, dim=0)
                            pred_class = torch.argmax(weighted_pred, dim=-1).item()
                        else:
                            pred_class = torch.argmax(pred[0], dim=-1).item()
                    else:
                        pred_class = torch.argmax(pred, dim=-1).item()
                else:
                    pred_class = int(pred)
                pred_classes_list.append(pred_class)

            pred_classes = np.array(pred_classes_list)
            pred_dist = np.zeros(self.config.nOut)

            for pred in pred_classes:
                if 0 <= pred < self.config.nOut:
                    pred_dist[pred] += 1

            if np.sum(pred_dist) == 0:
                return 0

            pred_dist = pred_dist / np.sum(pred_dist) + 1e-10

            label_dist = np.zeros(self.config.nOut)
            for label in labels:
                label_val = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
                if 0 <= label_val < self.config.nOut:
                    label_dist[label_val] += 1

            if np.sum(label_dist) == 0:
                return 0

            label_dist = label_dist / np.sum(label_dist) + 1e-10
            m_dist = (pred_dist + label_dist) / 2

            pred_dist = np.clip(pred_dist, 1e-10, 1.0)
            label_dist = np.clip(label_dist, 1e-10, 1.0)
            m_dist = np.clip(m_dist, 1e-10, 1.0)

            kl_p_m = np.sum(pred_dist * np.log(pred_dist / m_dist))
            kl_l_m = np.sum(label_dist * np.log(label_dist / m_dist))
            js_div = 0.5 * (kl_p_m + kl_l_m)

            if np.isnan(js_div) or np.isinf(js_div):
                return 0

            return js_div
        except Exception as e:
            print(f"Error calculating JS divergence: {e}")
            return 0

    def update_window_samples(self, pred, label):
        """Update sliding window of samples for drift detection."""
        try:
            self.window_samples.append((pred, label))
            if len(self.window_samples) > self.window_size:
                self.window_samples.pop(0)
        except Exception as e:
            print(f"Error updating window samples: {e}")

    def concept_drift_detect(self):
        """Detect concept drift based on alpha vector and JS divergence."""
        try:
            alpha_drift = torch.std(self.alpha_vector) <= self.config.theta

            if len(self.window_samples) >= self.window_size:
                preds = [s[0] for s in self.window_samples]
                labels = [s[1] for s in self.window_samples]
                js_div = self.calculate_js_divergence(preds, labels)

                if not hasattr(self, 'js_mean') or self.js_mean == 0:
                    self.js_mean = js_div
                else:
                    self.js_mean = 0.9 * self.js_mean + 0.1 * js_div

                self.js_drift = js_div > self.config.kly * self.js_mean if hasattr(self.config, 'kly') else False
            else:
                self.js_drift = False

            return alpha_drift
        except Exception as e:
            print(f"Error detecting concept drift: {e}")
            return False

    def add_neuron_to_last_layer(self):
        """Add a neuron to the last hidden layer."""
        try:
            last_hidden_idx = len(self.hidden_layers) - 1
            last_layer = self.hidden_layers[last_hidden_idx]
            in_features = last_layer.in_features
            out_features = last_layer.out_features

            new_layer = nn.Linear(in_features, out_features + 1)

            with torch.no_grad():
                new_layer.weight[:out_features] = last_layer.weight
                new_layer.bias[:out_features] = last_layer.bias
                nn.init.xavier_uniform_(new_layer.weight[out_features:])
                new_layer.bias[out_features:].zero_()

            self.hidden_layers[last_hidden_idx] = new_layer

            for i in range(len(self.output_layers)):
                if i + 1 == last_hidden_idx:
                    ol = self.output_layers[i]
                    new_ol = nn.Linear(out_features + 1, self.config.nOut)

                    with torch.no_grad():
                        new_ol.weight[:, :out_features] = ol.weight
                        new_ol.bias = ol.bias
                        nn.init.xavier_uniform_(new_ol.weight[:, out_features:])

                    self.output_layers[i] = new_ol

            self.window_samples = []
        except Exception as e:
            print(f"Error adding neuron: {e}")

    def add_hidden_layer(self):
        """Add a new hidden layer to the network."""
        try:
            last_hidden_dim = self.hidden_layers[-1].out_features

            new_hidden_layer = nn.Linear(last_hidden_dim, self.config.nHn)
            new_output_layer = nn.Linear(self.config.nHn, self.config.nOut)

            nn.init.xavier_uniform_(new_hidden_layer.weight)
            nn.init.zeros_(new_hidden_layer.bias)
            nn.init.xavier_uniform_(new_output_layer.weight)
            nn.init.zeros_(new_output_layer.bias)

            self.hidden_layers.append(new_hidden_layer)
            self.output_layers.append(new_output_layer)

            old_size = self.alpha_vector.size(0)
            device = self.alpha_vector.device

            new_alpha = torch.zeros(old_size + 1, requires_grad=False, device=device)
            new_gamma = torch.zeros(old_size + 1, requires_grad=False, device=device)
            new_v = torch.zeros(old_size + 1, requires_grad=False, device=device)

            new_alpha[:old_size] = self.alpha_vector.detach()
            new_gamma[:old_size] = self.gamma_vector.detach()
            new_v[:old_size] = self.v.detach()

            new_alpha[old_size] = 1.0 / (old_size + 1)
            new_gamma[old_size] = 1.0
            new_v[old_size] = 1.0
            new_alpha = new_alpha / torch.sum(new_alpha)

            self.alpha_vector = new_alpha
            self.gamma_vector = new_gamma
            self.v = new_v
            self.window_samples = []
        except Exception as e:
            print(f"Error adding hidden layer: {e}")

    def train_with_memory(self, optimizer):
        """Train the network using samples from memory pools."""
        try:
            # Train with local memory pool
            for sample, label, count in self.local_memory_pool:
                total_count = sum(s[2] for s in self.local_memory_pool)
                weight = count / total_count if total_count > 0 else 1.0

                pred = self(sample)
                label_scalar = label.item() if hasattr(label, 'item') else int(label)
                label_tensor = torch.tensor([label_scalar], dtype=torch.long, device=pred.device)

                weighted_preds = self.gamma_vector.unsqueeze(-1).unsqueeze(-1) * pred
                weighted_pred = torch.sum(weighted_preds, dim=0)
                loss = weight * F.cross_entropy(weighted_pred, label_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Train with global memory pool
            for sample, label in self.global_memory_pool:
                label_scalar = label.item() if hasattr(label, 'item') else int(label)
                label_tensor = torch.tensor([label_scalar], dtype=torch.long, device=sample.device)

                pred = self(sample)
                weighted_preds = self.gamma_vector.unsqueeze(-1).unsqueeze(-1) * pred
                weighted_pred = torch.sum(weighted_preds, dim=0)
                loss = F.cross_entropy(weighted_pred, label_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        except Exception as e:
            print(f"Error training with memory: {e}")

    def update_weights(self, losses):
        """Update alpha and gamma weight vectors based on losses."""
        try:
            if not self.config.del_da and not self.config.del_all:
                self.v = self.config.p * self.v + (1 - self.config.p) * torch.pow(self.config.beta, losses)
                new_alpha = torch.clamp(self.alpha_vector * self.v, min=self.config.smooth / self.alpha_vector.shape[0])
                self.alpha_vector = torch.divide(new_alpha, torch.sum(new_alpha))

            result = self.concept_drift_detect()
            js_drift = getattr(self, 'js_drift', False)

            if result:
                self.gamma_vector = torch.tensor(
                    [1. - torch.max(self.alpha_vector) for _ in range(self.alpha_vector.shape[0])],
                    device=self.alpha_vector.device)

                if js_drift:
                    self.add_hidden_layer()
                    self.local_memory_pool = []
            else:
                self.lw = torch.argmax(self.alpha_vector)
                indexs = torch.tensor(np.arange(self.alpha_vector.shape[0]), device=self.alpha_vector.device)
                l_best = torch.argmax(self.alpha_vector)
                split_index = l_best if l_best == self.alpha_vector.shape[0] else l_best + 1
                distance = torch.abs(indexs - l_best)
                dis_decay = distance / (torch.max(distance) + 1e-5)

                self.gamma_vector = torch.where(torch.lt(indexs, split_index),
                                                1. - torch.max(self.alpha_vector),
                                                torch.exp(-(dis_decay)))

                if js_drift:
                    self.add_neuron_to_last_layer()

            if self.config.del_pa or self.config.del_all:
                self.gamma_vector = self.alpha_vector.clone()
        except Exception as e:
            print(f"Error updating weights: {e}")

    def forward(self, x):
        """Forward pass through the network."""
        try:
            outputs = []
            out = F.relu(self.hidden_layers[0](x))

            CD_result = self.concept_drift_detect()

            if self.config.del_da or self.config.del_all:
                CD_result = True

            if not CD_result:
                l_best = torch.argmax(self.alpha_vector)
                split_index = l_best if l_best == self.alpha_vector.shape[0] else l_best + 1
                out_ = out

            for i, (hl, ol) in enumerate(zip(self.hidden_layers[1:], self.output_layers)):
                if not CD_result and i <= split_index:
                    out_ = F.relu(hl(out_))

                out = F.relu(hl(out))

                if not CD_result and i in [l_best, split_index]:
                    outputs.append(ol(out_))
                else:
                    outputs.append(ol(out))

                if CD_result:
                    out = out.detach()
                elif i > split_index or i < l_best:
                    out = out.detach()

            if len(outputs) == 0:
                outputs.append(self.output_layers[0](out))

            return torch.stack(outputs, dim=0)
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return torch.zeros((1, self.config.nOut))
