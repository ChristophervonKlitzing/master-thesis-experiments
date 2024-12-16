import torch
import torch.nn as nn
import torch.utils.data.dataloader
import zuko.flows as flows
import zuko


def random_rotate():
    roll = torch.rand(1,requires_grad=True) * 2 * torch.pi
    yaw = torch.rand(1,requires_grad=True) * 2 * torch.pi
    pitch = torch.rand(1,requires_grad=True) * 2 * torch.pi

    tensor_0 = torch.zeros(1)
    tensor_1 = torch.ones(1)

    RX = torch.stack([
                    torch.stack([tensor_1, tensor_0, tensor_0]),
                    torch.stack([tensor_0, torch.cos(roll), -torch.sin(roll)]),
                    torch.stack([tensor_0, torch.sin(roll), torch.cos(roll)])]).reshape(3,3)

    RY = torch.stack([
                    torch.stack([torch.cos(pitch), tensor_0, torch.sin(pitch)]),
                    torch.stack([tensor_0, tensor_1, tensor_0]),
                    torch.stack([-torch.sin(pitch), tensor_0, torch.cos(pitch)])]).reshape(3,3)

    RZ = torch.stack([
                    torch.stack([torch.cos(yaw), -torch.sin(yaw), tensor_0]),
                    torch.stack([torch.sin(yaw), torch.cos(yaw), tensor_0]),
                    torch.stack([tensor_0, tensor_0, tensor_1])]).reshape(3,3)

    return RX @ RY @ RZ

def _sum_rightmost(value: torch.Tensor, dim: int):
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


class SE3InvariantFlow(nn.Module):
    def __init__(self, point_dim: int, hidden_dim=128, num_layers=3, num_transforms: int = 3):
        super().__init__()
        self.flow = flows.MAF(point_dim, transforms=num_transforms, hidden_features=[hidden_dim] * num_layers)
        # self.flow = flows.NSF(point_dim, transforms=num_transforms, hidden_features = [hidden_dim] * num_layers)
        
    def flow_inverse(self, x):
        f: zuko.distributions.NormalizingFlow = self.flow()
        z, ladj = f.transform.call_and_ladj(x)
        neg_log_det = _sum_rightmost(ladj, -1)
        return z, neg_log_det
    
    def flow_forward(self, z) -> tuple[torch.Tensor, torch.Tensor]:
        f: zuko.distributions.NormalizingFlow = self.flow()
        x, ladj = f.transform.inv.call_and_ladj(z)
        log_det = _sum_rightmost(ladj, -1)
        return x, log_det
    
    def forward(self, z):
        return self.flow_inverse(z)

    def inverse(self, x):
        return self.flow_forward(x)


# Training loop
def train_flow(model: SE3InvariantFlow, dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, num_epochs: int):
    model.train()
    for epoch in range(num_epochs):
        for x in dataloader:
            x: torch.Tensor
            batch = x.shape[0]
            spatial_dim = 3
            n_nodes = x.shape[1] // spatial_dim

            translation = x.reshape(batch, n_nodes, spatial_dim).mean(-1) # shape (batch, 3)
            # print(translation.shape)
            
            # Forward pass

            translation = x.reshape(batch, n_nodes, spatial_dim).mean(1, keepdim=True)
            rand_translation = torch.randn((batch, 1, spatial_dim))
            x = (x.reshape(batch, n_nodes, spatial_dim) - translation + rand_translation).flatten(1)
            
            x_internal, log_det = model.forward(x)
            x_internal: torch.Tensor = x_internal.reshape(batch, n_nodes, spatial_dim)
            
            
            x_bar = x.reshape(batch, n_nodes, spatial_dim) - rand_translation
            x_internal_translated, log_det = model.forward(x_bar.flatten(1))
            x_internal_translated = x_internal_translated.reshape(batch, n_nodes, spatial_dim)

            reg = (x_internal_translated[:, 0, :]**2).mean() 
            reconstruction_loss = ((x_internal[:, 1:, :] - x_internal_translated[:, 1:, :])**2).mean() 
            
            loss = reconstruction_loss + 0.1 * reg

            # Backward and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")




def run(args):
    print("Test flow ability to learn SE(3) invariance")
    
    x = torch.tensor([1, 0, 0, 0, 0, 1]).reshape(1, 2, 3)
    print(x, x.T.shape)
    R = random_rotate()

    point_dim = 12  # For a point cloud with 100 points
    model = SE3InvariantFlow(point_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    samples = torch.vstack([
        torch.randn(100, point_dim),
        torch.randn(100, point_dim) * 0.5 + 3,
    ])
    # Dummy dataset (replace with real point cloud data)
    dataloader = torch.utils.data.DataLoader(
        samples, batch_size=100, shuffle=True
    )
    
    # Train the flow
    train_flow(model, dataloader, optimizer, num_epochs=200)

    x = model(torch.zeros([1, point_dim]))
    print(x[0].reshape(1, 4, 3))

    x = model(torch.zeros([1, point_dim]) + 3)
    print(x[0].reshape(1, 4, 3))

    print()
    x = torch.randn((1, point_dim))
    print(model(x)[0].reshape(1, 4, 3))
    print(model(x - 3)[0].reshape(1, 4, 3))

    print()
    y: torch.Tensor = model(x)[0].reshape(1, 4, 3)
    y[:, 0, :] = 0.0
    print(y)
    x_rec = model.inverse(y.flatten(1))[0]
    print(x_rec.reshape(1, 4, 3))
    print(x.reshape(1, 4, 3))
    print((x - x_rec).reshape(1, 4, 3))