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
        x1_flat = x1.flatten().detach().cpu().numpy()
        x2_flat = x2.flatten().detach().cpu().numpy()
        
        # 检查标准差为0的情况
        if np.std(x1_flat) == 0 or np.std(x2_flat) == 0:
            return 0
        
        corr = np.corrcoef(x1_flat, x2_flat)[0, 1]
        return corr if not np.isnan(corr) else 0

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

    def update_global_memory(self, x, y, error):
        """Update global memory pool based on prediction error."""
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

    def _extract_prediction_classes(self, preds):
        """Extract prediction classes from various prediction formats."""
        pred_classes = []
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
            pred_classes.append(pred_class)
        return np.array(pred_classes)

    def _calculate_distribution(self, classes):
        """Calculate probability distribution from class array."""
        dist = np.zeros(self.config.nOut)
        for cls in classes:
            if 0 <= cls < self.config.nOut:
                dist[cls] += 1
        
        if np.sum(dist) == 0:
            return None
        
        return dist / np.sum(dist) + 1e-10

    def calculate_js_divergence(self, preds, labels):
        """Calculate Jensen-Shannon divergence between predictions and labels."""
        if len(preds) < 2:
            return 0
        
        # Extract prediction classes
        pred_classes = self._extract_prediction_classes(preds)
        pred_dist = self._calculate_distribution(pred_classes)
        if pred_dist is None:
            return 0
        
        # Extract label classes
        label_classes = []
        for label in labels:
            label_val = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
            label_classes.append(label_val)
        
        label_dist = self._calculate_distribution(np.array(label_classes))
        if label_dist is None:
            return 0
        
        # Calculate JS divergence
        m_dist = (pred_dist + label_dist) / 2
        
        # Clip to avoid numerical issues
        pred_dist = np.clip(pred_dist, 1e-10, 1.0)
        label_dist = np.clip(label_dist, 1e-10, 1.0)
        m_dist = np.clip(m_dist, 1e-10, 1.0)
        
        kl_p_m = np.sum(pred_dist * np.log(pred_dist / m_dist))
        kl_l_m = np.sum(label_dist * np.log(label_dist / m_dist))
        js_div = 0.5 * (kl_p_m + kl_l_m)
        
        return js_div if not (np.isnan(js_div) or np.isinf(js_div)) else 0

    def update_window_samples(self, pred, label):
        """Update sliding window of samples for drift detection."""
        self.window_samples.append((pred, label))
        if len(self.window_samples) > self.window_size:
            self.window_samples.pop(0)

    def concept_drift_detect(self):
        """Detect concept drift based on alpha vector and JS divergence."""
        # Alpha drift detection
        alpha_drift = torch.std(self.alpha_vector) >= self.config.theta
        
        # JS divergence drift detection
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

    def add_neuron_to_last_layer(self):
        """Add a neuron to the last hidden layer."""
        last_hidden_idx = len(self.hidden_layers) - 1
        last_layer = self.hidden_layers[last_hidden_idx]
        in_features = last_layer.in_features
        out_features = last_layer.out_features
        
        # Create new layer with additional neuron
        new_layer = nn.Linear(in_features, out_features + 1)
        
        with torch.no_grad():
            new_layer.weight[:out_features] = last_layer.weight
            new_layer.bias[:out_features] = last_layer.bias
            nn.init.xavier_uniform_(new_layer.weight[out_features:])
            new_layer.bias[out_features:].zero_()
        
        self.hidden_layers[last_hidden_idx] = new_layer
        
        # Update corresponding output layer
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

    def add_hidden_layer(self):
        """Add a new hidden layer to the network."""
        last_hidden_dim = self.hidden_layers[-1].out_features
        
        # Create new layers
        new_hidden_layer = nn.Linear(last_hidden_dim, self.config.nHn)
        new_output_layer = nn.Linear(self.config.nHn, self.config.nOut)
        
        # Initialize weights
        nn.init.xavier_uniform_(new_hidden_layer.weight)
        nn.init.zeros_(new_hidden_layer.bias)
        nn.init.xavier_uniform_(new_output_layer.weight)
        nn.init.zeros_(new_output_layer.bias)
        
        self.hidden_layers.append(new_hidden_layer)
        self.output_layers.append(new_output_layer)
        
        # Update weight vectors
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

    def train_with_memory(self, optimizer):
        """Train the network using samples from memory pools."""
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

    def update_weights(self, losses):
        """Update alpha and gamma weight vectors based on losses."""
        # Update alpha vector
        if not self.config.del_da and not self.config.del_all:
            self.v = self.config.p * self.v + (1 - self.config.p) * torch.pow(self.config.beta, losses)
            new_alpha = torch.clamp(self.alpha_vector * self.v, min=self.config.smooth / self.alpha_vector.shape[0])
            self.alpha_vector = torch.divide(new_alpha, torch.sum(new_alpha))
        
        # Detect concept drift
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
        
        # Override gamma vector if specified in config
        if self.config.del_pa or self.config.del_all:
            self.gamma_vector = self.alpha_vector.clone()

    def forward(self, x):
        """Forward pass through the network."""
        outputs = []
        out = F.relu(self.hidden_layers[0](x))
        
        CD_result = self.concept_drift_detect()
        
        # Override concept drift result based on config
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
