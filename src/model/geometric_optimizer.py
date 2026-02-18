import torch
import torch.nn as nn
import torch.optim as optim

class GeometricCRISPROptimizer:
    """
    Optimizes CRISPR guides using Information Geometry.
    Minimizes thermodynamic length on the statistical manifold.

    Theory:
    Run Natural Gradient Descent on the Fisher Manifold.
    This is isomorphic to minimizing thermodynamic dissipation (Sivak & Crooks).

    Update: theta_{t+1} = theta_t - lr * G^{-1} * grad(L)
    """
    def __init__(self, model, lr=0.001, damping=1.0):
        self.model = model
        self.lr = lr
        self.damping = damping

    def compute_fisher_diag(self, inputs):
        """
        Computes the Diagonal Approximation of Empirical Fisher Information.
        G_diag(theta) = E [ grad(log p) ** 2 ]
        Returns a vector of size P.
        """
        fisher_sum = None

        # 1. Forward pass
        outputs = self.model(inputs)
        if isinstance(outputs, dict):
             outputs = outputs.get('on_target',
                       outputs.get('logits',
                       outputs.get('classification',
                       outputs.get('regression'))))

        # 2. Compute gradients for each sample
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Initialize cumulative squared gradients
        total_sq_grads = []
        for p in params:
            total_sq_grads.append(torch.zeros_like(p.view(-1)))

        total_sq_grads = torch.cat(total_sq_grads)

        # Iterate samples to save memory (accumulation)
        # Note: For strict Fisher, we need per-sample gradients squared.
        # F = 1/N * sum(g_i * g_i^T). Diagonal is 1/N * sum(g_i^2).

        log_probs = torch.log_softmax(outputs, dim=-1) if outputs.size(-1) > 1 else None

        for i in range(len(inputs)):
            self.model.zero_grad()

            if outputs.size(-1) == 1:
                # Regression: loss = 0.5 * (y_sample - pred)^2
                pred_val = outputs[i]
                y_sample = torch.normal(pred_val, 1.0)
                loss = 0.5 * (y_sample - pred_val)**2
            else:
                # Classification
                y_sample = torch.multinomial(torch.exp(log_probs[i]), 1)
                loss = -log_probs[i][y_sample]

            loss.backward(retain_graph=True)

            # Accumulate squared gradients
            current_grad_list = []
            for p in params:
                 if p.grad is not None:
                     current_grad_list.append(p.grad.view(-1))
                 else:
                     current_grad_list.append(torch.zeros_like(p.view(-1)))

            current_grad = torch.cat(current_grad_list)
            total_sq_grads += current_grad ** 2

        # 3. Average
        fisher_diag = total_sq_grads / len(inputs)

        return fisher_diag

    def step(self, inputs, targets):
        """
        Natural Gradient Step: theta_new = theta - lr * (G + lambda*I)^-1 * grad(L)
        Diagonal approx: theta_new = theta - lr * grad(L) / (diag(G) + lambda)
        """
        # 1. Standard Euclidean Gradient
        self.model.zero_grad()

        outputs = self.model(inputs)
        if isinstance(outputs, dict):
             outputs = outputs.get('on_target',
                       outputs.get('logits',
                       outputs.get('classification',
                       outputs.get('regression'))))

        if outputs.size(-1) == 1:
            loss = nn.MSELoss()(outputs, targets)
        else:
            loss = nn.CrossEntropyLoss()(outputs, targets)

        loss.backward()

        params = [p for p in self.model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        # Align euclidean_grad with G_diag (pad None with zeros)
        euclidean_grad_list = []
        for p in params:
            if p.grad is not None:
                euclidean_grad_list.append(p.grad.view(-1))
            else:
                euclidean_grad_list.append(torch.zeros_like(p.view(-1)))

        euclidean_grad = torch.cat(euclidean_grad_list)

        # 2. Fisher Diagonal
        G_diag = self.compute_fisher_diag(inputs)

        # 3. Natural Gradient (Diagonal Scaling)
        # element-wise division
        scaling = 1.0 / (G_diag + self.damping)
        natural_grad = euclidean_grad * scaling

        # 4. Update Weights
        idx = 0
        with torch.no_grad():
            for p in params:
                sz = p.numel()
                # Extract update chunk for this param (always exists due to padding)
                update = natural_grad[idx:idx+sz].view(p.shape)

                # Apply update (if grad existed, update is non-zero. If padded, update is zero)
                if p.grad is not None:
                    p.data -= self.lr * update

                # Critical: Always advance index to keep alignment
                idx += sz

        return loss.item()
