import torch
import torch.nn as nn


class MultiModalModel(nn.Module):

    def __init__(
        self,
        img_encoder,
        txt_encoder,
        dim=256,
        num_classes=23,
        use_prompt=True
    ):
        super().__init__()

        self.use_prompt = use_prompt

        # =========================
        # Encoders
        # =========================
        self.img_encoder = img_encoder
        self.txt_encoder = txt_encoder

        # =========================
        # Missing Tokens
        # =========================
        self.m_img = nn.Parameter(torch.randn(dim))
        self.m_txt = nn.Parameter(torch.randn(dim))

        # =========================
        # Gated Prompt
        # feature-aware + mask-aware
        # =========================

        gate_input_dim = dim * 2 + 2

        self.img_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

        self.txt_gate = nn.Sequential(
            nn.Linear(gate_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

        # =========================
        # Fusion
        # =========================
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, 256),
            nn.ReLU(),
        )

        # =========================
        # Classifier
        # =========================
        self.classifier = nn.Linear(256, num_classes)

        # =========================
        # Better initialization
        # =========================
        nn.init.constant_(self.img_gate[-1].bias, 1.0)
        nn.init.constant_(self.txt_gate[-1].bias, 1.0)

    def forward(self, image, text_inputs, mask):

        B = mask.size(0)
        device = mask.device

        # ==================================================
        # 1️⃣ Image Feature
        # ==================================================

        z_img = torch.zeros(
            B,
            self.m_img.size(0),
            device=device
        )

        img_idx = mask[:, 0] == 1

        if img_idx.sum() > 0:
            z_img[img_idx] = self.img_encoder(
                image[img_idx]
            )

        z_img[~img_idx] = self.m_img.unsqueeze(0)

        # ==================================================
        # 2️⃣ Text Feature
        # ==================================================

        z_txt = torch.zeros(
            B,
            self.m_txt.size(0),
            device=device
        )

        txt_idx = mask[:, 1] == 1

        if txt_idx.sum() > 0:
            z_txt[txt_idx] = self.txt_encoder(
                text_inputs["input_ids"][txt_idx],
                text_inputs["attention_mask"][txt_idx]
            )

        z_txt[~txt_idx] = self.m_txt.unsqueeze(0)

        # ==================================================
        # 3️⃣ Gated Prompt
        # ==================================================

        if self.use_prompt:

            # feature-aware + missing-aware
            gate_input = torch.cat([
                z_img.detach(),
                z_txt.detach(),
                mask.float()
            ], dim=-1)

            # separate gates
            img_gate = torch.sigmoid(
                self.img_gate(gate_input)
            )

            txt_gate = torch.sigmoid(
                self.txt_gate(gate_input)
            )

            # gated modulation
            z_img = z_img * img_gate
            z_txt = z_txt * txt_gate

        # ==================================================
        # 4️⃣ Fusion
        # ==================================================

        z = torch.cat([z_img, z_txt], dim=-1)

        z = self.fusion(z)

        # ==================================================
        # 5️⃣ Classification
        # ==================================================

        out = self.classifier(z)

        return out