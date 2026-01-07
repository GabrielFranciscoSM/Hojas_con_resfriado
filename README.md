# Detección y Clasificación de Patógenos Foliares mediante Visión por Computador

**Proyecto para la asignatura de Visión por Computador (VC) del Grado en Ingeniería Informática de la Universidad de Granada (UGR).**

---

## Autores

*   Gabriel Francisco Sánchez Muñoz
*   Germán Rodríguez Vidal
*   Miguel Ángel Moreno Castro
*   Pablo García Bas

---

## 1. Motivación y Problema

La detección temprana de enfermedades en plantas es crucial para la agricultura y la botánica. Sin embargo, muchas enfermedades foliares presentan síntomas visualmente similares, lo que dificulta un diagnóstico preciso. Un análisis aislado de la lesión, sin conocer la especie de la planta, puede llevar a conclusiones erróneas.

Este proyecto busca resolver esta ambigüedad desarrollando un sistema automatizado que no solo identifique la enfermedad, sino que primero determine la especie de la hoja analizada para contextualizar el diagnóstico.

---

## 2. Objetivos

*   **Principal:** Desarrollar un modelo de Deep Learning capaz de clasificar la especie de una planta, detectar las lesiones en sus hojas y clasificar el patógeno causante de la enfermedad con alta precisión.
*   **Secundarios:**
    *   Investigar y comparar la efectividad de diferentes arquitecturas de redes neuronales para cada una de las tareas (clasificación y detección).
    *   Crear un pipeline funcional que integre los diferentes modelos en una única secuencia de análisis.
    *   Evaluar el rendimiento del sistema utilizando métricas estándar en visión por computador (Accuracy, Precision, Recall, mAP).
    *   Documentar el proceso de desarrollo, los desafíos encontrados y los resultados obtenidos.

---

## 3. Metodología Propuesta

El sistema propuesto consiste en un pipeline de tres etapas que procesa una imagen de una hoja:

1.  **Clasificación de Especie:** Una Red Neuronal Convolucional (CNN) inicial recibe la imagen completa de la hoja y la clasifica para determinar la especie de la planta (ej. tomate, patata, manzano).
2.  **Detección de Lesiones:** Un modelo de detección de objetos, como **YOLO (You Only Look Once)**, analiza la imagen para localizar y dibujar cajas delimitadoras (bounding boxes) alrededor de las áreas que presentan síntomas de enfermedad.
3.  **Clasificación del Patógeno:** Las regiones de interés (los recortes de las cajas delimitadoras) generadas por YOLO son procesadas por una segunda CNN, especializada en clasificar el tipo de patógeno (ej. mildiu, oídio, roya) o determinar si el tejido es sano.

---

## 4. Datasets

La viabilidad del proyecto depende de la disponibilidad de datos etiquetados para las tres tareas. Se han identificado los siguientes datasets como puntos de partida:

*   **[New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data):** Contiene un gran volumen de imágenes (aprox. 87,000) de hojas de diferentes especies con diversas enfermedades. Ideal para las etapas de clasificación.
*   **[Plant Leaf Diseases Dataset](https://www.kaggle.com/datasets/nirmalsankalana/plant-diseases-training-dataset):** Otro recurso valioso para la clasificación de enfermedades.

*   **[Leaf disease segmentation dataset](https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset/data):** Dataset con máscaras para la segmentación.

*   **[Apple Tree Leaf Disease Segmentation Dataset](https://www.scidb.cn/en/detail?dataSetId=0e1f57004db842f99668d82183afd578):** Segmentación de hojas de manzana
*   **[Págnia con varios datasets](https://universe.roboflow.com/search?q=disease+-+v2+release+class%3Aleaf+object+detection):** Página web que contiene diferentes datasets para la segmentación de enfermedades de hojas.

### Datasets finales

*   **[Manzanas](https://www.scidb.cn/en/detail?dataSetId=0e1f57004db842f99668d82183afd578):** 
    * imágenes: 1641
    * Clases: 5
        *   **Alternaria leaf spot**: 278 imágenes
        *   **Brown spot**: 215 imágenes
        *   **Gray spot**: 395 imágenes
        *   **Healthy leaf**: 409 imágenes
        *   **Rust**: 344 imágenes
    
*   **[Tomates](https://universe.roboflow.com/hs1111/tomatoes-ddzvv):** 
    * imágenes: 3649
    *   Clases (2-4): 
        *   **Early blight** (1792 imágenes) (5290 patches):
            *   Early Blight: 554 imágenes
            *   Early_blight: 995
            *   Tomato - Early Blight: 185
            *   Tomato Early Blight: 58
        *   **late blight** (1551 imágenes) (3055 patches):
            *   late blight: 554
            *   late_blight: 1
            *   Late_blight: 996
        *   Leaf blight: 216
        *   Disease: 92 imágenes


*  **[Rosas](https://universe.roboflow.com/rose-leaf-diseases/rose-leaf-diseases):**
   * imagenes: 2725
   *   Clases: (4):
       *   **Black Spot** (5565 patches)
       *   **Powdery Mildew**(7346 patches)
       *   **Normal**(1598)
       *   Downy Mildew(1479)
       
* **[Patatas](https://app.roboflow.com/germanrv/potatoes_leaf-diseases/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)**
   * imagenes: 812
   *   Clases: (3)
       * early bright (18069 patches)
       * late bright   (1379 patches)
       * healthy (364 patches)
  


---

## 5. Tecnologías y Entorno

*   **Lenguaje:** Python
*   **Librerías Principales:** TensorFlow/Keras o PyTorch, OpenCV, Scikit-learn, Pandas, Matplotlib.
*   **Entorno de desarrollo:** Google Colab Pro (aprovechando sus GPUs para el entrenamiento).
*   **Control de versiones:** Git y GitHub.

---

## 6. Interesting Links

* **[YOLO from scratch](https://github.com/williamcfrancis/YOLOv3-Object-Detection-from-Scratch)**

---

## 7. TODOS

*   Datasets
    *   [ ] Elegir dataset adecuado, lo suficientemente grande y con suficientes ejemplos de plantas y enfermedades "similares"
    *   [ ] Estudiar si hay que preprocesar / aumentar el dataset
    *   [ ] Documentar Dataset
*   Modelo Clasificación
    *   [ ] Estudiar / Elegir Modelo de clasificación inicial del pipeline
*   Object detection
    *   [ ] Estudiar modelos de detección de objetoas
    *   [ ] Estudiar YOLO (diferentes versiones y/o implementación from scratch)
*   Finetuning
    *   [ ] Estudiar necesidad y factibilidad de hacer finetining para los modelos (los 3)
*   Integración
    *   [ ] Diseñar el pipeline con los diferentes modelos
    *   [ ] Testear y afinar los modelos
*   Documentación
