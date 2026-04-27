# Clasificación de Preguntas Acusatorias en Contratación Pública

Proyecto de Machine Learning para clasificar preguntas de procesos de contratación pública como **acusatorias** o **no acusatorias** usando técnicas de procesamiento de lenguaje natural en español.

El trabajo compara dos enfoques:

1. **Pipeline A:** TF-IDF + Regresión Logística regularizada.
2. **Pipeline B:** Sentence embeddings + StandardScaler + SVM con kernel RBF.

El notebook principal del repositorio es:

```text
proyectoML.ipynb
```

En la carpeta local de desarrollo también se trabajó una versión llamada:

```text
proyectoML_Final_Corregido.ipynb
```

La versión subida al repositorio corresponde a esa versión final corregida.

---

## Objetivo del Proyecto

El objetivo es construir y evaluar modelos capaces de detectar preguntas con tono acusatorio dentro de un conjunto de preguntas realizadas en procesos de contratación pública.

Este problema es relevante porque una pregunta acusatoria puede señalar inconformidades, sospechas de direccionamiento, falta de transparencia o posibles problemas en un proceso de contratación. Detectarlas automáticamente puede ayudar a priorizar revisión humana y apoyar procesos de seguimiento institucional.

El problema se plantea como una tarea de **clasificación binaria de texto**:

- `0`: pregunta no acusatoria.
- `1`: pregunta acusatoria.

---

## Dataset

El dataset usado es `dataset.xlsx`, con 5005 filas originales.

Columnas principales:

| Columna | Descripción |
|---|---|
| `contract_id` | Identificador del contrato o proceso |
| `pregunta_id` | Identificador de la pregunta |
| `pregunta` | Texto original de la pregunta |
| `sum_pregunta_isAcusatoria` | Suma de votos de anotadores |
| `final_pregunta_isAcusatoria` | Label final binario |

Distribución original de clases:

| Clase | Cantidad | Porcentaje |
|---|---:|---:|
| No acusatoria | 4858 | 97.06% |
| Acusatoria | 147 | 2.94% |

El dataset está fuertemente desbalanceado, con una razón aproximada de **33:1**. Por eso no se usa accuracy como única métrica.

---

## Limpieza y Deduplicación

Antes de entrenar los modelos se revisaron duplicados en la columna `pregunta`.

Resultados:

| Revisión | Resultado |
|---|---:|
| Filas originales | 5005 |
| Filas tras eliminar duplicados | 4858 |
| Duplicados removidos | 147 |
| Duplicados con labels contradictorios | 0 |

La **deduplicación** consiste en eliminar preguntas repetidas antes de hacer el train/test split. Esto evita fuga de información: si una pregunta idéntica aparece en entrenamiento y prueba, el modelo podría memorizarla y las métricas quedarían infladas.

Se deduplicó usando el texto original de `pregunta`, no el texto limpio. Esto evita favorecer artificialmente al pipeline TF-IDF y mantiene una comparación más justa con el modelo de embeddings.

---

## Preprocesamiento de Texto

Para el Pipeline A se creó la columna `pregunta_clean`, aplicando:

- conversión a minúsculas;
- eliminación de caracteres especiales;
- conservación de tildes, `ñ`, números y espacios;
- remoción de stopwords en español;
- conservación de palabras importantes para tono acusatorio.

Se conservaron palabras como:

```text
no, ni, nunca, jamás, tampoco, sin, cómo, cuándo, qué, cuál, debe, puede, pero, aunque
```

Estas palabras se mantienen porque pueden cambiar el sentido de una pregunta y aportar señal acusatoria. Por ejemplo, no es igual una pregunta neutral que una pregunta que contiene negación, sospecha o exigencia.

El split usado fue:

```text
80% entrenamiento
20% prueba
```

Con `stratify=y` para mantener la proporción de clases en train y test.

---

## Validación Cruzada

La evaluación interna usa:

```text
RepeatedStratifiedKFold
10 folds × 10 repeticiones = 100 evaluaciones
```

Se eligió esta configuración porque la rúbrica recomendaba repeated k-fold cross-validation con 10 repeticiones y 10 folds cuando el costo computacional lo permitiera.

La métrica principal de optimización fue:

```text
AUC-ROC
```

AUC-ROC se eligió porque mide la capacidad de separar clases en distintos umbrales, lo cual es más apropiado que accuracy en un problema desbalanceado.

---

## Pipeline A: TF-IDF + Regresión Logística

El primer enfoque usa un pipeline de sklearn:

```python
Pipeline([
    ("tfidf", TfidfVectorizer(...)),
    ("clf", LogisticRegression(...))
])
```

### Feature Extraction

`TfidfVectorizer` convierte cada pregunta limpia en un vector numérico. Cada dimensión representa una palabra o bigrama ponderado por TF-IDF.

Parámetros principales:

| Parámetro | Valor | Justificación |
|---|---|---|
| `ngram_range` | `(1, 2)` | Usa unigramas y bigramas |
| `min_df` | `2` | Elimina términos que aparecen una sola vez |
| `max_df` | `0.95` | Elimina términos demasiado frecuentes |
| `max_features` | optimizado | Controla tamaño del vocabulario |

### Feature Selection

La selección de features ocurre dentro de TF-IDF y se refuerza con regularización:

- `min_df=2` elimina términos demasiado raros.
- `max_df=0.95` elimina términos demasiado comunes.
- `max_features` controla cuántas features se conservan.
- `C` en Regresión Logística controla la regularización.

Además, se analizaron los coeficientes del modelo para identificar qué palabras y bigramas tienen más peso en la decisión.

Vocabulario final seleccionado:

| Tipo | Cantidad |
|---|---:|
| Unigramas | 5989 |
| Bigramas | 6337 |
| Total | 12326 |

Principales features asociadas a preguntas acusatorias:

```text
no
proceso
direccionado
sercop
marcas
ley
participación
direccionamiento
transparentes
```

Esto apoya la conclusión de que la señal discriminativa del problema es principalmente **léxica**.

### Optimización

Se usó `GridSearchCV` con:

```python
C = [0.001, 0.01, 0.1, 1, 10, 100]
max_features = [5000, 10000, None]
```

Mejores parámetros:

```python
clf__C = 1
tfidf__max_features = None
```

Resultado en validación cruzada:

| Métrica | Media ± std |
|---|---:|
| AUC-ROC | 0.9369 ± 0.0271 |
| F1-Macro | 0.7060 ± 0.0633 |
| Precision macro | 0.6933 ± 0.0639 |
| Recall macro | 0.7297 ± 0.0764 |
| Accuracy | 0.9629 ± 0.0090 |

---

## Pipeline B: Embeddings + SVM RBF

El segundo enfoque usa embeddings generados con:

```python
SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
```

Cada pregunta se transforma en un vector denso de 384 dimensiones.

Luego se usa un pipeline de sklearn:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", class_weight="balanced"))
])
```

### Decisiones de Diseño

| Componente | Justificación |
|---|---|
| Sentence embeddings | Capturan similitud semántica y contexto |
| Texto original | El transformer aprovecha estructura completa de la frase |
| StandardScaler | Escala las dimensiones antes del SVM |
| SVM RBF | Permite fronteras no lineales |
| `class_weight="balanced"` | Compensa el desbalance de clases |
| `decision_function` | Permite AUC-ROC sin usar `probability=True` |

### Optimización

Se usó `GridSearchCV` con:

```python
C = [1, 10, 100]
gamma = ["scale", "auto", 0.1]
```

Mejores parámetros:

```python
clf__C = 1
clf__gamma = "scale"
```

Resultado en validación cruzada:

| Métrica | Media ± std |
|---|---:|
| AUC-ROC | 0.9134 ± 0.0388 |
| F1-Macro | 0.7099 ± 0.0552 |
| Precision macro | 0.6897 ± 0.0524 |
| Recall macro | 0.7414 ± 0.0674 |
| Accuracy | 0.9621 ± 0.0086 |

---

## Visualización t-SNE

Se aplicó t-SNE sobre el espacio de embeddings, no sobre las predicciones del modelo.

El t-SNE:

- reduce los embeddings de 384 dimensiones a 2 dimensiones;
- permite visualizar si las clases se separan naturalmente;
- colorea cada punto usando el label real.

Interpretación:

El gráfico muestra que los puntos acusatorios no forman un cluster claramente separado. Esto sugiere que los embeddings agrupan las preguntas más por similitud semántica general que por tono acusatorio.

Por esa razón, para Pipeline A se prefirió interpretar coeficientes de Regresión Logística en lugar de hacer t-SNE sobre TF-IDF.

---

## Comparación Estadística

Se compararon los 100 scores de AUC-ROC de ambos modelos usando un test de Wilcoxon pareado.

Hipótesis:

- H0: ambos modelos tienen rendimiento equivalente.
- H1: existe diferencia entre los modelos.

Resultado:

| Estadístico | Valor |
|---|---:|
| W | 954.5000 |
| p-value | 0.000000 |

Como `p-value < 0.05`, se rechaza H0. La diferencia es estadísticamente significativa y Pipeline A obtiene mayor AUC-ROC promedio.

---

## Evaluación Final en Test Set

El test set contiene 972 muestras nunca vistas durante la validación cruzada.

| Métrica | Pipeline A | Pipeline B | Mejor |
|---|---:|---:|---|
| Accuracy | 0.9650 | 0.9609 | A |
| AUC-ROC | 0.9642 | 0.9582 | A |
| Average Precision | 0.4580 | 0.4222 | A |
| Precision macro | 0.7155 | 0.6881 | A |
| Precision weighted | 0.9729 | 0.9683 | A |
| Recall macro | 0.8149 | 0.7626 | A |
| Recall weighted | 0.9650 | 0.9609 | A |
| F1 score macro avg | 0.7548 | 0.7184 | A |
| F1 score weighted avg | 0.9683 | 0.9641 | A |
| AUC-ROC macro avg | 0.9642 | 0.9582 | A |
| AUC-ROC micro avg | 0.9924 | 0.9935 | B |

Pipeline A gana en la mayoría de métricas relevantes para este problema.

---

## Matriz de Confusión

Pipeline A:

|  | Pred no acusatoria | Pred acusatoria |
|---|---:|---:|
| Real no acusatoria | 919 | 24 |
| Real acusatoria | 10 | 19 |

Pipeline B:

|  | Pred no acusatoria | Pred acusatoria |
|---|---:|---:|
| Real no acusatoria | 918 | 25 |
| Real acusatoria | 13 | 16 |

Pipeline A detecta más preguntas acusatorias correctamente:

- Pipeline A: 19 verdaderos positivos.
- Pipeline B: 16 verdaderos positivos.

---

## Curva de Aprendizaje

Pipeline A mostró:

| Métrica | Valor |
|---|---:|
| AUC-ROC train | 0.9999 |
| AUC-ROC validación | 0.9391 |
| Gap train-validación | 0.0608 |

Esto indica **overfitting leve**. El modelo aprende casi perfecto en entrenamiento, pero mantiene buen desempeño en validación y test. No hay evidencia de underfitting.

---

## Conclusión Principal

El mejor modelo es:

```text
Pipeline A: TF-IDF + Regresión Logística regularizada
```

La razón principal es que el problema parece depender más de señales léxicas de tono acusatorio que de separación semántica general. Palabras como `no`, `direccionado`, `marcas`, `ley` y `direccionamiento` aportan información directa para identificar preguntas acusatorias.

Los embeddings son útiles, pero en este dataset agrupan más por tema general de contratación pública que por intención o tono acusatorio.

---

## Cómo Ejecutar el Proyecto

### 1. Clonar el repositorio

```bash
git clone https://github.com/rosvjames/machineLearningProyecto.git
cd machineLearningProyecto
```

### 2. Instalar dependencias

El notebook verifica e instala paquetes faltantes, pero también se pueden instalar manualmente:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy openpyxl nltk sentence-transformers
```

### 3. Abrir el notebook

```bash
jupyter notebook proyectoML.ipynb
```

o abrirlo desde VS Code/Jupyter Lab.

### 4. Ejecutar todas las celdas

El notebook descarga/verifica dependencias, carga `dataset.xlsx`, realiza EDA, entrena modelos, optimiza hiperparámetros y muestra resultados finales.

Nota: la ejecución completa puede tardar porque usa repeated k-fold cross-validation 10x10 y modelos con embeddings.

---

## Estructura del Repositorio

```text
.
├── dataset.xlsx
├── proyectoML.ipynb
├── README.md
└── .gitignore
```

---

## Métricas Reportadas

Como el proyecto es de clasificación, se reportan:

- Accuracy
- AUC-ROC
- AUC-ROC macro avg
- AUC-ROC micro avg
- Average Precision
- Precision macro
- Precision weighted
- Recall macro
- Recall weighted
- F1 score macro avg
- F1 score weighted avg
- Curva ROC
- Curva Precision-Recall
- Matriz de confusión

Métricas de regresión como MSE, RMSE y R-squared no aplican porque el target no es continuo.

La distorsión de clustering tampoco aplica porque no se trata de un problema no supervisado.

---

## Referencias Principales

- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR, 2011.
- Reimers y Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, EMNLP, 2019.
- Manning, Raghavan y Schütze, *Introduction to Information Retrieval*, Cambridge University Press, 2008.
- Cortes y Vapnik, *Support-vector networks*, Machine Learning, 1995.
- van der Maaten y Hinton, *Visualizing data using t-SNE*, JMLR, 2008.
- Wilcoxon, *Individual comparisons by ranking methods*, Biometrics Bulletin, 1945.
- Demšar, *Statistical comparisons of classifiers over multiple data sets*, JMLR, 2006.

