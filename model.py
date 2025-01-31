import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorDeep(nn.Module):
    def __init__(self, input_dim, h_dim, flag_dropout, k):
        super(GeneratorDeep, self).__init__()
        self.flag_dropout = flag_dropout

        # Shared representation layers
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2) if flag_dropout else nn.Identity(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2) if flag_dropout else nn.Identity()
        )

        # Treatment-specific layers
        self.treatment_encoder = nn.Sequential(
            nn.Linear(h_dim + 1, h_dim),  # +1 for treatment indicator
            nn.ReLU(),
            nn.Dropout(p=0.2) if flag_dropout else nn.Identity()
        )

        # Outcome-specific layers for treated and control
        self.treated_decoder = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2) if flag_dropout else nn.Identity(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

        self.control_decoder = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2) if flag_dropout else nn.Identity(),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.shared_encoder, self.treatment_encoder, 
                      self.treated_decoder, self.control_decoder]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x, t, y):
        if t.dim() == 3:
            t = t.squeeze(2)
        if t.dim() == 1:
            t = t.unsqueeze(1)

        # Shared feature extraction
        shared_features = self.shared_encoder(x)
        
        # Treatment-aware encoding
        treatment_features = self.treatment_encoder(
            torch.cat([shared_features, t], dim=1)
        )

        # Generate potential outcomes
        treated_outcome = self.treated_decoder(treatment_features)
        control_outcome = self.control_decoder(treatment_features)

        return torch.cat([control_outcome, treated_outcome], dim=1)

class Discriminator(nn.Module):
    def __init__(self, input_dim, h_dim, flag_dropout):
        super(Discriminator, self).__init__()
        self.flag_dropout = flag_dropout

        self.fc1 = nn.Linear(input_dim + 2, h_dim)  # +2 for y0 and y1
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2_1 = nn.Linear(h_dim, h_dim)
        self.dp2_1 = nn.Dropout(p=0.2)
        self.fc2_2 = nn.Linear(h_dim, h_dim)
        self.dp2_2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.dp2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(h_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc2_1, self.fc2_2, self.fc3]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, t, y, hat_y):
        if t.dim() == 3:
            t = t.squeeze(2)
        if y.dim() == 3:
            y = y.squeeze(2)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)

        input0 = (1. - t) * y + t * hat_y[:, 0].unsqueeze(1)
        input1 = t * y + (1. - t) * hat_y[:, 1].unsqueeze(1)
        inputs = torch.cat([x, input0, input1], dim=1)

        if self.flag_dropout:
            h1 = self.dp1(F.relu(self.fc1(inputs)))
            h2_1 = self.dp2_1(F.relu(self.fc2_1(h1)))
            h2_2 = self.dp2_2(F.relu(self.fc2_2(h2_1)))
            h2 = self.dp2(F.relu(self.fc2(h2_2)))
        else:
            h1 = F.relu(self.fc1(inputs))
            h2_1 = F.relu(self.fc2_1(h1))
            h2_2 = F.relu(self.fc2_2(h2_1))
            h2 = F.relu(self.fc2(h2_2))

        return self.fc3(h2)

class InferenceNetDeep(nn.Module):
    def __init__(self, input_dim, h_dim, flag_dropout, k):
        super(InferenceNetDeep, self).__init__()
        self.flag_dropout = flag_dropout

        self.fc1 = nn.Linear(input_dim, h_dim)
        self.dp1 = nn.Dropout(p=0.2)

        self.layers = nn.ModuleList()
        for _ in range(k-3):
            layer = nn.Linear(h_dim, h_dim)
            self.layers.append(layer)
            self.layers.append(nn.Dropout(p=0.2))

        self.fc31 = nn.Linear(h_dim, h_dim)
        self.fc32 = nn.Linear(h_dim, 1)

        self.fc41 = nn.Linear(h_dim, h_dim)
        self.fc42 = nn.Linear(h_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in [self.fc1, self.fc31, self.fc32, self.fc41, self.fc42]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        if self.flag_dropout:
            h = self.dp1(h)

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                h = F.relu(layer(h))
            else:
                h = layer(h)

        h31 = F.relu(self.fc31(h))
        logit1 = self.fc32(h31)
        y_hat_1 = torch.sigmoid(logit1)

        h41 = F.relu(self.fc41(h))
        logit2 = self.fc42(h41)
        y_hat_2 = torch.sigmoid(logit2)

        return torch.cat([y_hat_1, y_hat_2], dim=1)