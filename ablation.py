import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.manifold import TSNE
import time
from tqdm import tqdm
import argparse


class ChannelAttention(nn.Module):
    """Channel attention mechanism for feature refinement"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important regions"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class SparseAttention(nn.Module):
    """Sparse Attention module that combines channel and spatial attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)

        # Store attention values for analysis
        self.attention_values = None

    def forward(self, x):
        attention = self.channel_attention(x)
        self.attention_values = attention  # Store for analysis
        x = x * attention
        return x


class LocalFeatureAttention(nn.Module):
    """Local Feature Attention module for fine-grained feature refinement"""
    def __init__(self, channels):
        super().__init__()
        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Apply channel attention first
        x = x * self.channel_attention(x)
        # Then apply spatial attention
        x = x * self.spatial_attention(x)
        return x


class DSAN(nn.Module):
    """
    Dual Stream Attention Network for Facial Emotion Recognition

    Architecture:
    - GFE-AN: Global Feature Extraction with Attention Network
    - MFF-AN: Multi-scale Feature Fusion with Attention Network
    - Classification layer
    """
    def __init__(self, num_classes=7, pretrained=True, alpha=0.4, using_gfe=True, using_mff=True,
                 use_sa=True, use_lfa=True, F_value=4, reduction_ratio=2):
        super().__init__()

        # Configuration parameters
        self.alpha = alpha  # Balance parameter between GFE-AN and MFF-AN
        self.using_gfe = using_gfe  # Use GFE-AN stream
        self.using_mff = using_mff  # Use MFF-AN stream
        self.use_sa = use_sa  # Use Sparse Attention in GFE-AN
        self.use_lfa = use_lfa  # Use Local Feature Attention in MFF-AN
        self.F_value = F_value  # Number of local features in MFF-AN

        # Use ResNet18 as backbone for feature extraction
        resnet = models.resnet18(pretrained=pretrained)

        # Common layers for both streams
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # GFE-AN Stream (Global Feature Extraction with Attention Network)
        if self.using_gfe:
            self.gfe_layer1 = resnet.layer1
            self.gfe_attention1 = SparseAttention(64, reduction_ratio) if use_sa else nn.Identity()

            self.gfe_layer2 = resnet.layer2
            self.gfe_attention2 = SparseAttention(128, reduction_ratio) if use_sa else nn.Identity()

            self.gfe_layer3 = resnet.layer3
            self.gfe_layer4 = resnet.layer4

            # GFE classification head
            self.gfe_avgpool = nn.AdaptiveAvgPool2d(1)
            self.gfe_fc = nn.Linear(512, num_classes)

        # MFF-AN Stream (Multi-scale Feature Fusion with Attention Network)
        if self.using_mff:
            # Create separate layers for MFF-AN
            self.mff_layer1 = nn.Sequential(*list(resnet.layer1.children()))
            self.mff_layer2 = nn.Sequential(*list(resnet.layer2.children()))
            self.mff_layer3 = nn.Sequential(*list(resnet.layer3.children()))
            self.mff_layer4 = nn.Sequential(*list(resnet.layer4.children()))

            # Local feature attention blocks
            if use_lfa:
                self.mff_attention1 = LocalFeatureAttention(256)
                self.mff_attention2 = LocalFeatureAttention(512)
            else:
                self.mff_attention1 = nn.Identity()
                self.mff_attention2 = nn.Identity()

            # MFF classification head
            self.mff_avgpool = nn.AdaptiveAvgPool2d(1)
            self.mff_fc = nn.Linear(512, num_classes)

        # Storage for feature maps and attention maps
        self.gfe_features = None
        self.mff_features = None
        self.attention_maps = []

    def get_features_for_loss(self):
        """Returns features suitable for loss calculation"""
        if self.using_gfe and self.gfe_features is not None:
            # Average pooling to get a feature vector
            return F.adaptive_avg_pool2d(self.gfe_features, 1).squeeze(-1).squeeze(-1)
        elif self.using_mff and self.mff_features is not None:
            # Average pooling to get a feature vector
            return F.adaptive_avg_pool2d(self.mff_features, 1).squeeze(-1).squeeze(-1)
        else:
            return None

    def forward(self, x):
        # Reset storage
        self.attention_maps = []
        self.gfe_features = None
        self.mff_features = None

        # Common feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        gfe_out = None
        mff_out = None

        # GFE-AN Stream
        if self.using_gfe:
            gfe = self.gfe_layer1(x)
            gfe = self.gfe_attention1(gfe)

            gfe = self.gfe_layer2(gfe)
            gfe = self.gfe_attention2(gfe)

            gfe = self.gfe_layer3(gfe)
            gfe = self.gfe_layer4(gfe)

            # Store feature maps for visualization and loss calculation
            self.gfe_features = gfe  # Don't detach here to allow gradient flow

            # Create a pooled version for classification
            gfe_pooled = self.gfe_avgpool(gfe)
            gfe_pooled = torch.flatten(gfe_pooled, 1)
            gfe_out = self.gfe_fc(gfe_pooled)

        # MFF-AN Stream
        if self.using_mff:
            mff = self.mff_layer1(x)
            mff = self.mff_layer2(mff)
            mff = self.mff_layer3(mff)

            # Apply LFA to layer3 output
            mff = self.mff_attention1(mff)

            mff = self.mff_layer4(mff)

            # Apply LFA to layer4 output
            mff = self.mff_attention2(mff)

            # Store feature maps for visualization and loss calculation
            self.mff_features = mff  # Don't detach here to allow gradient flow

            # Create a pooled version for classification
            mff_pooled = self.mff_avgpool(mff)
            mff_pooled = torch.flatten(mff_pooled, 1)
            mff_out = self.mff_fc(mff_pooled)

        # Combine outputs based on alpha parameter
        if self.using_gfe and self.using_mff:
            output = self.alpha * gfe_out + (1 - self.alpha) * mff_out
        elif self.using_gfe:
            output = gfe_out
        elif self.using_mff:
            output = mff_out
        else:
            raise ValueError("At least one stream (GFE-AN or MFF-AN) must be used")

        return output

    def get_attention_sparsity(self):
        """Calculate the sparsity of attention maps"""
        if not self.using_gfe or not self.use_sa:
            return -1

        # Get channel attention values from GFE-AN
        if hasattr(self.gfe_attention1, 'attention_values') and self.gfe_attention1.attention_values is not None:
            attention = self.gfe_attention1.attention_values
            # Calculate proportion of near-zero values (using small threshold)
            threshold = 0.01
            total_elements = attention.numel()
            zero_elements = torch.sum(attention < threshold).item()
            sparsity = zero_elements / total_elements
            return sparsity
        return -1

    def extract_features(self, x):
        """Extract features from both streams for visualization"""
        _ = self.forward(x)  # Forward pass to populate features
        
        # Make a copy to avoid modifying the original features
        gfe_features = self.gfe_features.detach().clone() if self.gfe_features is not None else None
        mff_features = self.mff_features.detach().clone() if self.mff_features is not None else None
        
        return gfe_features, mff_features


class RAFDBFolderDataset(Dataset):
    """
    RAF-DB dataset loader for folder-based structure

    The RAF-DB dataset contains 7 emotion categories mapped to folder numbers:
    1: Surprise, 2: Fear, 3: Disgust, 4: Happiness, 5: Sadness, 6: Anger, 7: Neutral
    """
    def __init__(self, root_dir, split='test', transform=None):
        """
        Args:
            root_dir (string): Root directory of the RAF-DB dataset.
            split (string): 'train' or 'test' split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.samples = []

        # Check if directory exists
        if not os.path.exists(self.root_dir):
            raise RuntimeError(f"Dataset directory not found: {self.root_dir}")

        # Class mapping based on RAF-DB folder numbering
        self.class_to_idx = {
            '1': 0,  # Surprise
            '2': 1,  # Fear
            '3': 2,  # Disgust
            '4': 3,  # Happiness
            '5': 4,  # Sadness
            '6': 5,  # Anger
            '7': 6,  # Neutral
        }

        # Load all samples from the directory structure
        for class_folder in sorted(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_folder)
            if os.path.isdir(class_path) and class_folder in self.class_to_idx:
                class_idx = self.class_to_idx[class_folder]
                img_files = os.listdir(class_path)
                print(f"Class {class_folder} ({self.get_class_name(class_idx)}): {len(img_files)} images")
                for img_file in img_files:
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_path, img_file), class_idx))

    def get_class_name(self, class_idx):
        """Get emotion name from class index"""
        emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
        return emotion_labels[class_idx]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image and the same label
            placeholder = torch.zeros(3, 224, 224) if self.transform else Image.new('RGB', (224, 224))
            return placeholder, label


class CenterLoss(nn.Module):
    """Center loss for intra-class feature distance minimization"""
    def __init__(self, num_classes=7, feat_dim=512, device='cuda'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        # Centers for each class
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers.index_select(0, labels)

        # Calculate L2 distance between features and their corresponding class centers
        loss = torch.sum((features - centers_batch)**2) / batch_size
        return loss


class FeatureRecalibrationLoss(nn.Module):
    """
    Feature Recalibration Loss (LFR) as described in the paper
    Combines:
    - LFR_CE: Feature recalibration cross-entropy loss
    - LFR_C: Feature recalibration center loss
    - LCCS: Class center separation loss
    """
    def __init__(self, num_classes=7, feat_dim=512, lambda1=0.01, lambda2=0.01, device='cuda'):
        super(FeatureRecalibrationLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda1 = lambda1  # Weight for LFR_C
        self.lambda2 = lambda2  # Weight for LCCS
        self.device = device

        # Create centers parameter for center loss components
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

        # Regular cross entropy loss
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, features, logits, labels):
        batch_size = features.size(0)

        # 1. LFR_CE: Feature recalibration cross-entropy loss
        # Standard cross-entropy loss
        lfr_ce = self.ce_loss(logits, labels)

        # 2. LFR_C: Feature recalibration center loss
        centers_batch = self.centers.index_select(0, labels)
        lfr_c = torch.sum((features - centers_batch)**2) / batch_size

        # 3. LCCS: Class center separation loss
        # Compute pairwise distances between class centers
        center_distances = torch.cdist(self.centers, self.centers, p=2)

        # Create a mask to ignore the diagonal (distance to self is 0)
        mask = 1 - torch.eye(self.num_classes, device=self.device)

        # Compute the mean of non-diagonal distances
        lccs = -torch.sum(center_distances * mask) / (self.num_classes * (self.num_classes - 1))

        # Combine all losses with their weights
        total_loss = lfr_ce + self.lambda1 * lfr_c + self.lambda2 * lccs

        return total_loss, lfr_ce, lfr_c, lccs


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model(model, test_loader, device, criterion=None):
    """Test the model on the test dataset"""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    class_correct = [0] * 7
    class_total = [0] * 7
    emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

    confusion_matrix = torch.zeros(7, 7)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Calculate loss if criterion is provided
            if criterion is not None:
                if isinstance(criterion, FeatureRecalibrationLoss):
                    if hasattr(model, 'get_features_for_loss'):
                        features = model.get_features_for_loss()
                        if features is not None:
                            loss, _, _, _ = criterion(features, outputs, labels)
                        else:
                            # Fallback to standard CE if features aren't available
                            loss = nn.CrossEntropyLoss()(outputs, labels)
                    else:
                        # For models without the get_features_for_loss method
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                else:
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)

            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

                # Update confusion matrix
                confusion_matrix[label][pred] += 1

    # Calculate overall accuracy
    accuracy = 100 * correct / total

    # Calculate average loss if criterion was provided
    avg_loss = running_loss / total if criterion is not None else 0.0

    print(f'Test Accuracy: {accuracy:.2f}%')
    if criterion is not None:
        print(f'Test Loss: {avg_loss:.4f}')

    # Calculate per-class accuracy
    print('\nPer-class accuracy:')
    for i in range(7):
        class_acc = 100 * class_correct[i] / max(class_total[i], 1)
        print(f'{emotion_labels[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})')

    # Calculate F1 score for each class
    print('\nPer-class F1 scores:')
    f1_scores = []
    for i in range(7):
        # Calculate precision and recall
        tp = confusion_matrix[i][i].item()
        fp = confusion_matrix[:, i].sum().item() - tp
        fn = confusion_matrix[i, :].sum().item() - tp

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        # Calculate F1 score
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        f1_scores.append(f1)
        print(f'{emotion_labels[i]}: {f1:.4f}')

    # Calculate mean F1 score
    mean_f1 = sum(f1_scores) / len(f1_scores)
    print(f'\nMean F1 Score: {mean_f1:.4f}')

    return accuracy, mean_f1, confusion_matrix, avg_loss


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs=25, save_path=None):
    """Train the model and evaluate on test set"""
    best_acc = 0.0
    training_stats = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'time_per_epoch': []
    }

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss based on criterion type
            if isinstance(criterion, FeatureRecalibrationLoss):
                if hasattr(model, 'get_features_for_loss'):
                    features = model.get_features_for_loss()
                    if features is not None:
                        loss, _, _, _ = criterion(features, outputs, labels)
                    else:
                        # Fallback to standard CE if features aren't available
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                else:
                    # For models without the get_features_for_loss method
                    loss = nn.CrossEntropyLoss()(outputs, labels)
            else:
                # Standard loss function
                loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100 * correct / total

        # Test phase
        test_acc, _, _, test_loss = test_model(model, test_loader, device, criterion)

        # Step the scheduler
        if scheduler:
            scheduler.step()

        # Record statistics
        training_stats['train_loss'].append(epoch_loss)
        training_stats['train_acc'].append(epoch_acc)
        training_stats['test_loss'].append(test_loss)
        training_stats['test_acc'].append(test_acc)
        training_stats['time_per_epoch'].append(time.time() - epoch_start)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}%')
        print(f'Time: {time.time() - epoch_start:.2f}s')

        # Save best model
        if save_path and test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f'Saved model with acc: {best_acc:.2f}%')

    print(f'Best test accuracy: {best_acc:.2f}%')
    return training_stats


def visualize_training_history(history):
    """Visualize training history"""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['test_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['test_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()


def evaluate_feature_distribution(model, test_loader, device):
    """Evaluate feature distributions with t-SNE visualization"""
    model.eval()
    emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

    # Initialize empty lists to store features and labels
    all_gfe_features = []
    all_mff_features = []
    all_labels = []

    # Extract features from all test samples
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Extracting features"):
            images = images.to(device)

            # Forward pass to extract features
            gfe_features, mff_features = model.extract_features(images)

            if gfe_features is not None:
                # Convert to feature vectors
                gfe_features = F.adaptive_avg_pool2d(gfe_features, 1).squeeze(-1).squeeze(-1)
                all_gfe_features.append(gfe_features.cpu())

            if mff_features is not None:
                # Convert to feature vectors
                mff_features = F.adaptive_avg_pool2d(mff_features, 1).squeeze(-1).squeeze(-1)
                all_mff_features.append(mff_features.cpu())

            all_labels.append(labels)

    # Concatenate features and labels
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Create figure
    plt.figure(figsize=(20, 10))

    # Process GFE features if available
    if all_gfe_features:
        all_gfe_features = torch.cat(all_gfe_features, dim=0).numpy()

        # Apply t-SNE dimensionality reduction
        print("Running t-SNE on GFE features...")
        gfe_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(all_gfe_features)

        # Plot
        plt.subplot(1, 2, 1)
        for i, emotion in enumerate(emotion_labels):
            idx = all_labels == i
            plt.scatter(gfe_tsne[idx, 0], gfe_tsne[idx, 1], c=colors[i], label=emotion, alpha=0.7)

        plt.title('GFE-AN Feature Distribution')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()

    # Process MFF features if available
    if all_mff_features:
        all_mff_features = torch.cat(all_mff_features, dim=0).numpy()

        # Apply t-SNE dimensionality reduction
        print("Running t-SNE on MFF features...")
        mff_tsne = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(all_mff_features)

        # Plot
        plt.subplot(1, 2, 2)
        for i, emotion in enumerate(emotion_labels):
            idx = all_labels == i
            plt.scatter(mff_tsne[idx, 0], mff_tsne[idx, 1], c=colors[i], label=emotion, alpha=0.7)

        plt.title('MFF-AN Feature Distribution')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()

    plt.tight_layout()
    plt.savefig('feature_distribution.png')
    plt.close()

    print("Feature distribution visualization saved to feature_distribution.png")


def evaluate_alpha_parameter(test_loader, device, raf_db_root, model_path=None, pretrained=False):
    """Evaluate the impact of alpha parameter on model performance"""
    print("\nEvaluating alpha parameter impact:")

    alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    accuracies = []

    for alpha in alpha_values:
        print(f"Testing with alpha = {alpha:.1f}")

        # Create model with current alpha value
        model = DSAN(num_classes=7, pretrained=pretrained, alpha=alpha)
        model = model.to(device)

        # Load pretrained weights if provided
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model weights from {model_path}")

        # Test model
        accuracy, _, _, _ = test_model(model, test_loader, device)
        accuracies.append(accuracy)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, accuracies, 'o-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Alpha Value')
    plt.ylabel('Accuracy (%)')
    plt.title('Impact of Alpha Parameter on Model Performance')
    plt.xticks(alpha_values)
    plt.ylim(min(accuracies) - 2, max(accuracies) + 2)

    # Add text labels above points
    for i, acc in enumerate(accuracies):
        plt.text(alpha_values[i], acc + 0.5, f'{acc:.2f}%', ha='center')

    plt.savefig('alpha_parameter_evaluation.png')
    plt.close()

    print(f"Best alpha value: {alpha_values[np.argmax(accuracies)]:.1f} with accuracy: {max(accuracies):.2f}%")
    print("Alpha parameter evaluation saved to alpha_parameter_evaluation.png")


def evaluate_lambda_parameters(train_loader, test_loader, device, raf_db_root, num_epochs=5):
    """Evaluate the impact of lambda1 and lambda2 parameters on model performance"""
    print("\nEvaluating lambda parameters impact:")

    # Values to test for lambda1 (fixes lambda2 at 0.01)
    lambda1_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    lambda1_accuracies = []

    # Fixed lambda2 value
    fixed_lambda2 = 0.01

    # Test different lambda1 values
    for lambda1 in lambda1_values:
        print(f"Testing with lambda1 = {lambda1} and lambda2 = {fixed_lambda2}")

        # Create model
        model = DSAN(num_classes=7, pretrained=True, alpha=0.4)
        model = model.to(device)

        # Define loss function with current lambda values
        criterion = FeatureRecalibrationLoss(
            num_classes=7,
            feat_dim=512,
            lambda1=lambda1,
            lambda2=fixed_lambda2,
            device=device
        )

        # Define optimizer and scheduler

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Train model for a few epochs
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=num_epochs,
            save_path=None  # Don't save intermediary models during parameter search
        )

        # Record best accuracy
        lambda1_accuracies.append(max(history['test_acc']))

        # Values to test for lambda2 (fixes lambda1 at 0.01)
        lambda2_values = [0.0, 0.005, 0.01, 0.05, 0.1]
        lambda2_accuracies = []

        # Fixed lambda1 value
        fixed_lambda1 = 0.01

        # Test different lambda2 values
        for lambda2 in lambda2_values:
            print(f"Testing with lambda1 = {fixed_lambda1} and lambda2 = {lambda2}")

            # Create model
            model = DSAN(num_classes=7, pretrained=True, alpha=0.4)
            model = model.to(device)

            # Define loss function with current lambda values
            criterion = FeatureRecalibrationLoss(
                num_classes=7,
                feat_dim=512,
                lambda1=fixed_lambda1,
                lambda2=lambda2,
                device=device
            )

            # Define optimizer and scheduler
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

            # Train model for a few epochs
            history = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=num_epochs,
                save_path=None
            )

            # Record best accuracy
            lambda2_accuracies.append(max(history['test_acc']))

        # Plot results for lambda1
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.semilogx(lambda1_values, lambda1_accuracies, 'o-', linewidth=2)
        plt.grid(True)
        plt.xlabel('Lambda1 Value (log scale)')
        plt.ylabel('Accuracy (%)')
        plt.title('Impact of Lambda1 Parameter (with Lambda2=0.01)')

        # Add text labels above points
        for i, acc in enumerate(lambda1_accuracies):
            plt.text(lambda1_values[i], acc + 0.5, f'{acc:.2f}%', ha='center')

        # Plot results for lambda2
        plt.subplot(1, 2, 2)
        plt.plot(lambda2_values, lambda2_accuracies, 'o-', linewidth=2)
        plt.grid(True)
        plt.xlabel('Lambda2 Value')
        plt.ylabel('Accuracy (%)')
        plt.title('Impact of Lambda2 Parameter (with Lambda1=0.01)')

        # Add text labels above points
        for i, acc in enumerate(lambda2_accuracies):
            plt.text(lambda2_values[i], acc + 0.5, f'{acc:.2f}%', ha='center')

        plt.tight_layout()
        plt.savefig('lambda_parameters_evaluation.png')
        plt.close()

        print(f"Best lambda1 value: {lambda1_values[np.argmax(lambda1_accuracies)]} with accuracy: {max(lambda1_accuracies):.2f}%")
        print(f"Best lambda2 value: {lambda2_values[np.argmax(lambda2_accuracies)]} with accuracy: {max(lambda2_accuracies):.2f}%")
        print("Lambda parameters evaluation saved to lambda_parameters_evaluation.png")

def main(args):
    """Main function to train and evaluate the DSAN model"""
    # If no args provided, create default args
    if args is None:
        args = type('Args', (), {
            'data_dir': './data/rafdb/DATASET',
            'batch_size': 32,
            'epochs': 50,
            'lr': 0.01,
            'alpha': 0.4,
            'lambda1': 0.01,
            'lambda2': 0.01,
            'F_value': 4,
            'reduction_ratio': 2,
            'mode': 'train',
            'model_path': './dsan_model.pth',
            'no_gfe': False,
            'no_mff': False,
            'no_sa': False,
            'no_lfa': False
        })()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    try:
        print(f"Loading RAF-DB dataset from {args.data_dir}")
        train_dataset = RAFDBFolderDataset(root_dir=args.data_dir, split='train', transform=train_transform)
        test_dataset = RAFDBFolderDataset(root_dir=args.data_dir, split='test', transform=test_transform)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Create model
    model = DSAN(
        num_classes=7,
        pretrained=True,
        alpha=args.alpha,
        using_gfe=not args.no_gfe,
        using_mff=not args.no_mff,
        use_sa=not args.no_sa,
        use_lfa=not args.no_lfa,
        F_value=args.F_value,
        reduction_ratio=args.reduction_ratio
    )
    model = model.to(device)

    # Print model information
    print("\nModel Configuration:")
    print(f"Using GFE-AN: {not args.no_gfe}")
    print(f"Using MFF-AN: {not args.no_mff}")
    print(f"Using Sparse Attention: {not args.no_sa}")
    print(f"Using Local Feature Attention: {not args.no_lfa}")
    print(f"Alpha parameter: {args.alpha}")
    print(f"F value: {args.F_value}")
    print(f"Reduction ratio: {args.reduction_ratio}")
    print(f"Total parameters: {count_parameters(model):,}")

    # Define loss function
    criterion = FeatureRecalibrationLoss(
        num_classes=7,
        feat_dim=512,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        device=device
    )

    # Define optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # Mode selection
    if args.mode == 'train':
        print("\nTraining model...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.epochs,
            save_path=args.model_path
        )

        # Visualize training history
        visualize_training_history(history)

        # Evaluate feature distribution
        evaluate_feature_distribution(model, test_loader, device)

    elif args.mode == 'test':
        # Load pretrained model
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Loaded model from {args.model_path}")
        else:
            print(f"Model file {args.model_path} not found. Using untrained model.")

        # Test model
        accuracy, mean_f1, confusion_matrix, _ = test_model(model, test_loader, device, nn.CrossEntropyLoss())

        # Evaluate feature distribution
        evaluate_feature_distribution(model, test_loader, device)

        # Calculate sparsity
        if not args.no_gfe and not args.no_sa:
            sparsity = model.get_attention_sparsity()
            print(f"Average attention sparsity: {sparsity:.4f}")

    elif args.mode == 'evaluate_alpha':
        evaluate_alpha_parameter(test_loader, device, args.data_dir, args.model_path)

    elif args.mode == 'evaluate_lambda':
        evaluate_lambda_parameters(train_loader, test_loader, device, args.data_dir, num_epochs=5)

    elif args.mode == 'ablation':
        # Create different model configurations for ablation study
        print("\nPerforming ablation studies...")

        # Test baseline (ResNet-18 with max-pooling)
        baseline_model = models.resnet18(pretrained=True)
        baseline_model.fc = nn.Linear(512, 7)  # Change output layer for 7 emotions
        baseline_model = baseline_model.to(device)

        print("\nTesting baseline (ResNet-18):")
        baseline_acc, _, _, _ = test_model(baseline_model, test_loader, device, nn.CrossEntropyLoss())

        # Test with different configurations of DSAN
        configs = [
            {'name': 'HFE1 only', 'using_gfe': True, 'using_mff': False, 'use_sa': False, 'use_lfa': False},
            {'name': 'HFE1 + SA', 'using_gfe': True, 'using_mff': False, 'use_sa': True, 'use_lfa': False},
            {'name': 'HFE2 only', 'using_gfe': False, 'using_mff': True, 'use_sa': False, 'use_lfa': False},
            {'name': 'HFE2 + LFA', 'using_gfe': False, 'using_mff': True, 'use_sa': False, 'use_lfa': True},
            {'name': 'HFE1 + HFE2', 'using_gfe': True, 'using_mff': True, 'use_sa': False, 'use_lfa': False},
            {'name': 'HFE1 + HFE2 + LFA', 'using_gfe': True, 'using_mff': True, 'use_sa': False, 'use_lfa': True},
            {'name': 'HFE1 + SA + HFE2', 'using_gfe': True, 'using_mff': True, 'use_sa': True, 'use_lfa': False},
            {'name': 'Full DSAN', 'using_gfe': True, 'using_mff': True, 'use_sa': True, 'use_lfa': True}
        ]

        results = []
        for config in configs:
            print(f"\nTesting configuration: {config['name']}")
            test_model_instance = DSAN(
                num_classes=7,
                pretrained=True,
                alpha=args.alpha,
                using_gfe=config['using_gfe'],
                using_mff=config['using_mff'],
                use_sa=config['use_sa'],
                use_lfa=config['use_lfa'],
                F_value=args.F_value,
                reduction_ratio=args.reduction_ratio
            )
            test_model_instance = test_model_instance.to(device)

            # Test model
            acc, _, _, _ = test_model(test_model_instance, test_loader, device, nn.CrossEntropyLoss())
            results.append({'config': config['name'], 'accuracy': acc})

        # Print and plot ablation results
        print("\nAblation Study Results:")
        for result in results:
            print(f"{result['config']}: {result['accuracy']:.2f}%")

        # Plot results
        plt.figure(figsize=(12, 6))
        names = [r['config'] for r in results]
        accs = [r['accuracy'] for r in results]

        plt.bar(names, accs)
        plt.axhline(y=baseline_acc, color='r', linestyle='--', label=f'Baseline: {baseline_acc:.2f}%')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Accuracy (%)')
        plt.title('Ablation Study Results')
        plt.legend()
        plt.tight_layout()
        plt.savefig('ablation_results.png')
        plt.close()

        print("Ablation results saved to ablation_results.png")

    print("\nDone!")


if __name__ == "__main__":
    # Define default arguments
    args = type('Args', (), {
        'data_dir': './data/rafdb/DATASET',  # CHANGE THIS LINE to point to your dataset
        'batch_size': 32,
        'epochs': 50,
        'lr': 0.01,
        'alpha': 0.4,
        'lambda1': 0.01,
        'lambda2': 0.01,
        'F_value': 4,
        'reduction_ratio': 2,
        'mode': 'train',
        'model_path': './dsan_model.pth',
        'no_gfe': False,
        'no_mff': False,
        'no_sa': False,
        'no_lfa': False
    })()

    # Call main function with the arguments
    main(args)
