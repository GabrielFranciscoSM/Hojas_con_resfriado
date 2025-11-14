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

---

## 5. Tecnologías y Entorno

*   **Lenguaje:** Python
*   **Librerías Principales:** TensorFlow/Keras o PyTorch, OpenCV, Scikit-learn, Pandas, Matplotlib.
*   **Entorno de desarrollo:** Google Colab Pro (aprovechando sus GPUs para el entrenamiento).
*   **Control de versiones:** Git y GitHub.

---

## 6. Planificación (Cronograma a 2 meses)

*   **Semanas 1-2:**
    *   [ ] Búsqueda y análisis exhaustivo de datasets.
    *   [ ] Definición final del alcance del proyecto (ver "Consideraciones Clave").
    *   [ ] Preprocesamiento y limpieza de los datos.
*   **Semanas 3-4:**
    *   [ ] Desarrollo y entrenamiento del **Modelo 1 (Clasificación de Especie)**.
    *   [ ] Inicio del etiquetado de datos para YOLO (si es necesario).
*   **Semanas 5-6:**
    *   [ ] Desarrollo y entrenamiento del **Modelo 2 (Detección de Lesiones con YOLO)**.
    *   [ ] Desarrollo y entrenamiento del **Modelo 3 (Clasificación de Patógeno)**.
*   **Semanas 7-8:**
    *   [ ] Integración de los tres modelos en un pipeline único.
    *   [ ] Evaluación final del sistema completo.
    *   [ ] Redacción de la memoria y preparación de la presentación final.
