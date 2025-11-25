import torch


def log_softmax(z):
    z_max, _ = torch.max(z, dim=1, keepdim=True)
    z_shifted = z - z_max
    log_sum_exp = torch.log(
        torch.sum(torch.exp(z_shifted), dim=1, keepdim=True)
    )  # kq
    return z_shifted - log_sum_exp


def nll_loss(y_hat, y):
    log_probs = log_softmax(y_hat)
    return torch.mean(-log_probs[range(len(y)), y])
