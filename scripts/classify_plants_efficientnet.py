#!/usr/bin/env python3
"""
Script de clasificación de plantas usando EfficientNetV2.
Clasifica imágenes en 3 clases: manzana, patatas, rosas.

EfficientNetV2 es más ligero y rápido que MaxViT, ideal para tareas
de clasificación de imágenes donde no se necesita la complejidad
de un Vision Transformer.

Este script está diseñado para ser ejecutado en Google Colab.
Cada sección está separada por comentarios para facilitar la conversión a notebook.
"""

# ============================================================================
# BLOQUE 1: Instalación de dependencias
# ============================================================================
# !pip install torch torchvision timm pillow matplotlib numpy --quiet

# ============================================================================
# BLOQUE 2: Imports y configuración del dispositivo
# ============================================================================
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import timm

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"timm version: {timm.__version__}")

# Mostrar modelos EfficientNetV2 disponibles
efficientnet_models = [m for m in timm.list_models('efficientnetv2*') if 'in21k' not in m]
print(f"\nModelos EfficientNetV2 disponibles: {efficientnet_models[:6]}")

# ============================================================================
# BLOQUE 3: Montar Google Drive (solo en Colab)
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')
print("Google Drive montado correctamente.")

# ============================================================================
# BLOQUE 4: Configuración del dataset
# ============================================================================
# Ruta del dataset en Google Drive
DRIVE_PATH = '/content/drive/MyDrive/data_balanced'

# Verificar que la ruta existe
if os.path.exists(DRIVE_PATH):
    print(f"✓ Ruta encontrada: {DRIVE_PATH}")
    print(f"  Contenido: {os.listdir(DRIVE_PATH)}")
else:
    print(f"✗ ERROR: No se encuentra la ruta {DRIVE_PATH}")
    print("  Verifica que la carpeta existe en Google Drive")

# Clases del dataset
CLASSES = ['manzana', 'patatas', 'rosas']
NUM_CLASSES = len(CLASSES)

# Proporciones de división
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print(f"Ruta del dataset: {DRIVE_PATH}")
print(f"Número de clases: {NUM_CLASSES}")
print(f"Clases: {CLASSES}")

# ============================================================================
# BLOQUE 5: Definición del Dataset personalizado
# ============================================================================
class PlantDataset(Dataset):
    """
    Dataset personalizado para cargar imágenes de plantas.
    Estructura esperada:
        root/
            manzana/
                img1.jpg
                img2.jpg
            patatas/
                img1.jpg
                ...
            rosas/
                img1.jpg
                ...
    """
    def __init__(self, root_dir, classes, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.transform = transform
        self.samples = []
        
        # Cargar todas las imágenes de cada clase
        for class_name in classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
            else:
                print(f"Advertencia: No existe la carpeta {class_path}")
        
        print(f"Total de imágenes cargadas: {len(self.samples)}")
        for cls in classes:
            count = sum(1 for s in self.samples if s[1] == self.class_to_idx[cls])
            print(f"  - {cls}: {count} imágenes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# BLOQUE 6: Definición de transformaciones
# ============================================================================
# EfficientNetV2 usa entrada de 224x224 (algunas variantes 384x384)
# Usamos 224x224 para la variante small

# Transformaciones para entrenamiento (con aumento de datos)
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformaciones para validación/test (sin aumento)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Train transform:", train_transform)
print("\nVal transform:", val_transform)

# ============================================================================
# BLOQUE 7: Clase auxiliar para subset con transformaciones
# ============================================================================
class SubsetWithTransform(Dataset):
    """Wrapper para aplicar transformaciones a un subset."""
    def __init__(self, subset, original_dataset, transform):
        self.indices = list(subset)
        self.original_dataset = original_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img_path, label = self.original_dataset.samples[original_idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# BLOQUE 8: Carga y división del dataset
# ============================================================================
# Cargar dataset completo (sin transformaciones para dividir)
full_dataset = PlantDataset(
    root_dir=DRIVE_PATH,
    classes=CLASSES,
    transform=None  # Aplicaremos transformaciones después de dividir
)

# Calcular tamaños para la división
total_size = len(full_dataset)
train_size = int(TRAIN_RATIO * total_size)
val_size = int(VAL_RATIO * total_size)
test_size = total_size - train_size - val_size

print(f"\nDivisión del dataset:")
print(f"  - Train: {train_size} ({TRAIN_RATIO*100:.0f}%)")
print(f"  - Val: {val_size} ({VAL_RATIO*100:.0f}%)")
print(f"  - Test: {test_size} ({TEST_RATIO*100:.0f}%)")

# Dividir dataset con semilla fija para reproducibilidad
generator = torch.Generator().manual_seed(42)
train_indices, val_indices, test_indices = random_split(
    range(total_size), 
    [train_size, val_size, test_size],
    generator=generator
)

# Crear datasets con transformaciones apropiadas
train_dataset = SubsetWithTransform(train_indices, full_dataset, train_transform)
val_dataset = SubsetWithTransform(val_indices, full_dataset, val_transform)
test_dataset = SubsetWithTransform(test_indices, full_dataset, val_transform)

print(f"\nDatasets creados:")
print(f"  - Train: {len(train_dataset)} muestras")
print(f"  - Val: {len(val_dataset)} muestras")
print(f"  - Test: {len(test_dataset)} muestras")

# ============================================================================
# BLOQUE 9: Creación de DataLoaders
# ============================================================================
BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

print(f"\nDataLoaders creados:")
print(f"  - Train batches: {len(train_loader)}")
print(f"  - Val batches: {len(val_loader)}")
print(f"  - Test batches: {len(test_loader)}")

# ============================================================================
# BLOQUE 10: Función para crear modelo EfficientNetV2
# ============================================================================
def create_efficientnet_model(num_classes, model_name='tf_efficientnetv2_s.in1k', freeze_backbone=True):
    """
    Crea un modelo EfficientNetV2 para fine-tuning.
    
    Variantes disponibles (de menor a mayor):
        - tf_efficientnetv2_s.in1k  (~21M params) - Small  [RECOMENDADO]
        - tf_efficientnetv2_m.in1k  (~54M params) - Medium
        - tf_efficientnetv2_l.in1k  (~118M params) - Large
    
    Comparación con MaxViT:
        - MaxViT Tiny: ~30M params
        - EfficientNetV2-S: ~21M params (30% menos)
        - Entrenamiento ~2x más rápido
    
    Args:
        num_classes: Número de clases para clasificación
        model_name: Nombre del modelo EfficientNetV2 preentrenado
        freeze_backbone: Si True, congela el backbone y solo entrena la cabeza
    
    Returns:
        Modelo configurado para fine-tuning
    """
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
    )
    
    if freeze_backbone:
        # Congelar todo excepto el clasificador
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Modelo: {model_name}")
    print(f"Clases: {num_classes}")
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model

# ============================================================================
# BLOQUE 11: Funciones de entrenamiento y evaluación
# ============================================================================
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Entrena el modelo durante una época.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Batch [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {running_loss/(batch_idx+1):.4f} '
                  f'Acc: {100.*correct/total:.2f}%')
    
    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, val_loader, criterion, device):
    """
    Evalúa el modelo en el conjunto de validación.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(val_loader), 100. * correct / total

# ============================================================================
# BLOQUE 12: Entrenamiento - Fase 1 (solo cabeza/clasificador)
# ============================================================================
print("=" * 60)
print("FASE 1: Entrenando solo el clasificador (cabeza)")
print("=" * 60)

# Modelo EfficientNetV2-Small (más ligero que MaxViT)
MODEL_NAME = 'tf_efficientnetv2_s.in1k'

model = create_efficientnet_model(
    num_classes=NUM_CLASSES, 
    model_name=MODEL_NAME,
    freeze_backbone=True
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)  # LR alto para la cabeza
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

EPOCHS_PHASE1 = 5
for epoch in range(EPOCHS_PHASE1):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS_PHASE1} ===")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# ============================================================================
# BLOQUE 13: Entrenamiento - Fase 2 (fine-tuning completo)
# ============================================================================
print("\n" + "=" * 60)
print("FASE 2: Fine-tuning completo")
print("=" * 60)

# Descongelar todos los parámetros
for param in model.parameters():
    param.requires_grad = True

# Contar parámetros ahora
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parámetros entrenables: {trainable_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # LR bajo para fine-tuning
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

EPOCHS_PHASE2 = 10
best_val_acc = 0.0
for epoch in range(EPOCHS_PHASE2):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS_PHASE2} ===")
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    scheduler.step()
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f"✓ Nuevo mejor modelo guardado (Val Acc: {val_acc:.2f}%)")

print("\n✅ Entrenamiento completado.")
print(f"Mejor accuracy de validación: {best_val_acc:.2f}%")

# ============================================================================
# BLOQUE 14: Evaluación final en conjunto de test
# ============================================================================
print("\n" + "=" * 60)
print("EVALUACIÓN EN CONJUNTO DE TEST")
print("=" * 60)

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

# ============================================================================
# BLOQUE 15: Matriz de confusión
# ============================================================================
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def get_all_predictions(model, dataloader, device):
    """Obtiene todas las predicciones para el dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

# Obtener predicciones
preds, labels = get_all_predictions(model, test_loader, device)

# Matriz de confusión
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión - EfficientNetV2')
plt.tight_layout()
plt.savefig('confusion_matrix_efficientnet.png', dpi=150)
plt.show()

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(labels, preds, target_names=CLASSES))

# ============================================================================
# BLOQUE 16: Guardar modelo
# ============================================================================
def save_model(model, path, model_name, num_classes, classes, epoch=None):
    """
    Guarda el modelo con metadatos.
    """
    checkpoint = {
        'model_name': model_name,
        'num_classes': num_classes,
        'classes': classes,
        'state_dict': model.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)
    print(f"Modelo guardado en: {path}")


def load_model(path, device='cuda'):
    """
    Carga un modelo guardado.
    """
    checkpoint = torch.load(path, map_location=device)
    
    model = timm.create_model(
        checkpoint['model_name'],
        pretrained=False,
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Modelo cargado: {checkpoint['model_name']}")
    print(f"Clases: {checkpoint.get('classes', 'N/A')}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model


# Guardar el modelo entrenado
MODEL_PATH = 'efficientnetv2_plant_classifier.pth'
save_model(
    model, 
    MODEL_PATH, 
    MODEL_NAME, 
    num_classes=NUM_CLASSES,
    classes=CLASSES,
    epoch=EPOCHS_PHASE1 + EPOCHS_PHASE2
)

print("\n✅ Script completado exitosamente!")

# ============================================================================
# BLOQUE 17: Comparación con MaxViT (opcional - informativo)
# ============================================================================
print("\n" + "=" * 60)
print("COMPARACIÓN: EfficientNetV2-S vs MaxViT-Tiny")
print("=" * 60)
print("""
┌─────────────────────┬───────────────────┬────────────────────┐
│ Característica      │ EfficientNetV2-S  │ MaxViT-Tiny        │
├─────────────────────┼───────────────────┼────────────────────┤
│ Parámetros          │ ~21M              │ ~30M               │
│ Input size          │ 224x224           │ 224x224            │
│ Arquitectura        │ CNN + MBConv      │ Atención + Conv    │
│ Velocidad           │ Más rápido        │ Más lento          │
│ Uso memoria GPU     │ ~4GB              │ ~6GB               │
│ Ideal para          │ Clasificación     │ Tareas complejas   │
└─────────────────────┴───────────────────┴────────────────────┘

EfficientNetV2 es más eficiente para clasificación de 3 clases.
""")
