from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, n_layers, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        if n_layers == 1:
            layers = [
                nn.Linear(input_dim, output_dim),   
            ]
        else:
            layers = [
                nn.Linear(input_dim, hidden_dim)
            ]
            for _ in range(n_layers-2):
                layers.extend([
                    nn.ELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ])
            layers.extend([
                nn.Linear(hidden_dim, output_dim),
            ])
            
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)