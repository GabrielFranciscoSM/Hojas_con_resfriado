#!/usr/bin/env python3
"""
Script para preparar la carpeta de datos para clasificación.
Organiza las imágenes de manzana, patatas y rosas en una estructura limpia y balanceada:

data_balanced/
├── manzana/
│   ├── img1.jpg
│   └── ... (800 imágenes)
├── patatas/
│   ├── img1.jpg
│   └── ... (800 imágenes)
└── rosas/
    ├── img1.jpg
    └── ... (800 imágenes)
"""

import os
import shutil
import random
from pathlib import Path

# Configuración
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "data_balanced"
CLASSES = ["manzana", "patatas", "rosas"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Número máximo de imágenes por clase (se ajusta al mínimo disponible)
MAX_IMAGES_PER_CLASS = 800

# Semilla para reproducibilidad
random.seed(42)


def find_images(folder: Path) -> list:
    """Encuentra todas las imágenes en una carpeta recursivamente, excluyendo 'labels'."""
    images = []
    for root, dirs, files in os.walk(folder):
        # Excluir carpetas de labels
        if "labels" in root.lower():
            continue
        for file in files:
            if Path(file).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(Path(root) / file)
    return images


def copy_images_with_prefix(images: list, dest_folder: Path, prefix: str):
    """Copia imágenes a la carpeta destino con un prefijo para evitar colisiones."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    copied = 0
    for i, img_path in enumerate(images):
        new_name = f"{prefix}_{i:05d}{img_path.suffix.lower()}"
        dest_path = dest_folder / new_name
        shutil.copy2(img_path, dest_path)
        copied += 1
    return copied


def main():
    print("=" * 50)
    print("Preparando datos BALANCEADOS para clasificación")
    print("=" * 50)
    
    # Primero, contar imágenes por clase para determinar el mínimo
    class_images = {}
    for class_name in CLASSES:
        source_folder = DATA_DIR / class_name
        if source_folder.exists():
            class_images[class_name] = find_images(source_folder)
            print(f"{class_name}: {len(class_images[class_name])} imágenes encontradas")
        else:
            print(f"⚠️  Carpeta no encontrada: {source_folder}")
            class_images[class_name] = []
    
    # Determinar el número de imágenes a usar (mínimo entre clases y MAX)
    min_available = min(len(imgs) for imgs in class_images.values() if imgs)
    images_per_class = min(min_available, MAX_IMAGES_PER_CLASS)
    
    print(f"\n→ Se usarán {images_per_class} imágenes por clase")
    
    # Limpiar carpeta de salida si existe
    if OUTPUT_DIR.exists():
        print(f"\nEliminando carpeta existente: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Carpeta de salida creada: {OUTPUT_DIR}\n")
    
    total_images = 0
    
    for class_name in CLASSES:
        dest_folder = OUTPUT_DIR / class_name
        
        # Seleccionar imágenes aleatorias
        images = class_images[class_name]
        if len(images) > images_per_class:
            selected = random.sample(images, images_per_class)
        else:
            selected = images
        
        print(f"Procesando: {class_name}")
        print(f"  - Imágenes seleccionadas: {len(selected)}/{len(images)}")
        
        copied = copy_images_with_prefix(selected, dest_folder, class_name)
        print(f"  - Imágenes copiadas: {copied}")
        total_images += copied
    
    print("\n" + "=" * 50)
    print(f"✅ Proceso completado!")
    print(f"   Total de imágenes: {total_images}")
    print(f"   Imágenes por clase: {images_per_class}")
    print(f"   Ubicación: {OUTPUT_DIR}")
    print("=" * 50)
    
    # Mostrar resumen
    print("\nResumen por clase:")
    for class_name in CLASSES:
        class_folder = OUTPUT_DIR / class_name
        if class_folder.exists():
            count = len(list(class_folder.glob("*")))
            print(f"  - {class_name}: {count} imágenes")


if __name__ == "__main__":
    main()
