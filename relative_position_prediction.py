import torch
import torch.nn as nn
import torch.nn.functional as F
from deepul_helper.tasks.rotation import NetworkInNetwork, AlexNet


class RelativePositionPrediction(nn.Module):
    """
    Relative Position Prediction: Prédire la position relative d'un patch
    par rapport à un patch de référence (8 positions possibles).
    
    Inspiré de: "Unsupervised Visual Representation Learning by Context Prediction"
    Doersch et al., ICCV 2015
    """
    metrics = ['Loss', 'Acc1']
    metrics_fmt = [':.4e', ':6.2f']

    def __init__(self, dataset, n_classes):
        super().__init__()
        
        # Architecture selon le dataset
        if dataset == 'cifar10':
            self.model = NetworkInNetwork()
            self.latent_dim = 192 * 8 * 8
            self.feat_layer = 'conv2'
        elif 'imagenet' in dataset:
            self.model = AlexNet()
            self.latent_dim = 256 * 13 * 13
            self.feat_layer = 'conv5'
        else:
            raise Exception(f'Unsupported dataset: {dataset}')
        
        self.dataset = dataset
        self.n_classes = n_classes
        
        # Tête de classification pour prédire la position (8 classes)
        self.position_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.latent_dim * 2, 512),  # 2 patches concatenés
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 8)  # 8 positions possibles
        )

    def extract_patches(self, images):
        """
        Extrait 9 patches d'une image (1 centre + 8 positions autour)
        Retourne: (query_patch, context_patches, positions)
        """
        batch_size = images.shape[0]
        _, _, H, W = images.shape
        
        # Taille des patches (1/3 de l'image)
        patch_h, patch_w = H // 3, W // 3
        
        # Extraire le patch central (query)
        center_h, center_w = H // 2, W // 2
        query = images[:, :, 
                      center_h - patch_h//2 : center_h + patch_h//2,
                      center_w - patch_w//2 : center_w + patch_w//2]
        
        # Positions des 8 patches autour du centre
        # 0 1 2
        # 3 X 4
        # 5 6 7
        positions_coords = [
            (0, 0), (0, patch_w), (0, 2*patch_w),           # top row
            (patch_h, 0), (patch_h, 2*patch_w),             # middle row (skip center)
            (2*patch_h, 0), (2*patch_h, patch_w), (2*patch_h, 2*patch_w)  # bottom row
        ]
        
        context_patches = []
        for h_start, w_start in positions_coords:
            patch = images[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w]
            context_patches.append(patch)
        
        return query, context_patches

    def construct_classifier(self):
        """Construit un classifieur linéaire pour la classification downstream"""
        if self.dataset == 'cifar10':
            classifier = nn.Sequential(
                nn.Flatten(),
                nn.BatchNorm1d(self.latent_dim, affine=False),
                nn.Linear(self.latent_dim, self.n_classes)
            )
        elif 'imagenet' in self.dataset:
            classifier = nn.Sequential(
                nn.AdaptiveMaxPool2d((6, 6)),
                nn.BatchNorm2d(256, affine=False),
                nn.Flatten(),
                nn.Linear(256 * 6 * 6, self.n_classes)
            )
        else:
            raise Exception(f'Unsupported dataset: {self.dataset}')
        return classifier

    def forward(self, images):
        """
        Forward pass pour l'entraînement sur la tâche de prédiction de position
        """
        batch_size = images.shape[0]
        
        # Extraire patches
        query, context_patches = self.extract_patches(images)
        
        # Pour chaque image, créer 8 paires (query, context_i)
        all_queries = query.repeat(8, 1, 1, 1)  # (8*B, C, H, W)
        all_contexts = torch.cat(context_patches, dim=0)  # (8*B, C, H, W)
        
        # Encoder les patches
        query_features = self.model(all_queries, out_feat_keys=(self.feat_layer,))
        context_features = self.model(all_contexts, out_feat_keys=(self.feat_layer,))
        
        # Aplatir les features
        query_flat = torch.flatten(query_features, 1)
        context_flat = torch.flatten(context_features, 1)
        
        # Concaténer query + context
        combined = torch.cat([query_flat, context_flat], dim=1)
        
        # Prédire la position
        logits = self.position_classifier(combined)
        
        # Créer les targets (positions 0-7 répétées pour chaque batch)
        targets = torch.arange(8).long().repeat(batch_size)
        targets = targets.to(images.device)
        
        # Loss et accuracy
        loss = F.cross_entropy(logits, targets)
        pred = logits.argmax(dim=-1)
        correct = pred.eq(targets).float().sum()
        acc = correct / targets.shape[0] * 100.
        
        # Retourner les features du premier batch pour downstream tasks
        return dict(Loss=loss, Acc1=acc), query_features[:batch_size]

    def encode(self, images):
        """Encode les images pour extraction de features"""
        zs = self.model(images, out_feat_keys=(self.feat_layer,))
        return zs


if __name__ == '__main__':
    # Test du modèle
    print("Testing RelativePositionPrediction...")
    model = RelativePositionPrediction(dataset='cifar10', n_classes=10)
    print(f"✓ Model created successfully")
    print(f"✓ Latent dim: {model.latent_dim}")
    print(f"✓ Feature layer: {model.feat_layer}")
    
    # Test avec des données factices
    dummy_images = torch.randn(4, 3, 32, 32)
    metrics, features = model(dummy_images)
    print(f"✓ Forward pass successful")
    print(f"  Loss: {metrics['Loss']:.4f}")
    print(f"  Acc1: {metrics['Acc1']:.2f}%")
    print(f"  Features shape: {features.shape}")