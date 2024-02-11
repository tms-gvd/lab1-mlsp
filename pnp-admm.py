import torch
import torch.nn.functional as F

class PnP_ADMM:
    def __init__(self, max_iter=100, rho=1.0, beta=1.0, tolerance=1e-4):
        self.max_iter = max_iter
        self.rho = rho
        self.beta = beta
        self.tolerance = tolerance
        self.x = None
        self.z = None
        self.u = None
        self.cost_history = []

    def fit(self, A, y, proximal_operator, data_fidelity_term, regularization_term):
        self.x = torch.zeros_like(y, requires_grad=True)
        self.z = torch.zeros_like(y, requires_grad=True)
        self.u = torch.zeros_like(y, requires_grad=True)

        for iteration in range(self.max_iter):
            # Update x using the proximal operator
            self.update_x(proximal_operator)

            # Update z
            self.update_z(A, y, data_fidelity_term, regularization_term)

            # Update u
            self.update_u()

            # Compute and store the cost function value
            cost = self.compute_cost(A, y, data_fidelity_term, regularization_term)
            self.cost_history.append(cost.item())

            # Check convergence
            if iteration > 0 and abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                break

        return self.x.detach().numpy(), self.cost_history

    def update_x(self, proximal_operator):
        self.x = proximal_operator(self.z - self.u, self.beta)

    def update_z(self, A, y, data_fidelity_term, regularization_term):
        self.z = F.adaptive_avg_pool2d(self.z, (A.shape[2], A.shape[3]))  # Ensure z has the same size as A
        self.z = data_fidelity_term(A, self.x, y) + regularization_term(self.x) + self.u

    def update_u(self):
        self.u = self.u + self.x - self.z

    def compute_cost(self, A, y, data_fidelity_term, regularization_term):
        cost = data_fidelity_term(A, self.x, y) + regularization_term(self.x)
        return cost

# Example usage
if __name__ == "__main__":
    # Define a simple proximal operator (e.g., L1 soft-thresholding)
    def proximal_operator(z, beta):
        return torch.nn.functional.softshrink(z, beta)

    # Define a simple data fidelity term and regularization term
    def data_fidelity_term(A, x, y):
        return 0.5 * torch.norm(A @ x - y)**2

    def regularization_term(x):
        return torch.norm(x, p=1)

    # Generate synthetic data
    torch.manual_seed(42)
    A = torch.randn(10, 10, 64, 64)
    true_x = torch.randn(10, 10, 64, 64)
    y = A @ true_x + 0.1 * torch.randn(10, 10, 64, 64)

    # Set PnP-ADMM parameters
    max_iter = 50
    rho = 1.0
    beta = 0.1
    tolerance = 1e-4

    # Initialize and fit the PnP-ADMM model
    pnp_admm = PnP_ADMM(max_iter=max_iter, rho=rho, beta=beta, tolerance=tolerance)
    result_x, cost_history = pnp_admm.fit(A, y, proximal_operator, data_fidelity_term, regularization_term)

    print("Final Result (x):", result_x.shape)
    print("Final Cost:", cost_history[-1])
