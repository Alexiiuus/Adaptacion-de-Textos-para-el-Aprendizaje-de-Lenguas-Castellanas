# Informe Preliminar

## **Adaptación de Textos para el Aprendizaje de Lenguas Castellanas**

## Resumen
En este proyecto buscamos: a partir de un texto en Castellano y un nivel de Aprendizaje del MCER (A1, A2, B1, B2, C1, C2), aplicar un modelo de lenguajes para transcribir el texto al nivel deseado sin perder su significado original.

## Hipótesis de trabajo
- Todo texto es categorizable en algún nivel de aprendizaje MCER (A1, A2, B1, B2, C1, C2).
- Es posible reescribir cualquier texto en distintos niveles sin modificar su significado original.

## Objetivos preliminares
1. Conseguir un Dataset con textos en Castellano etiquetados por nivel (A1, A2, B1, B2, C1, C2).
   - Intentamos conseguir un Dataset de estas características, pero no tuvimos éxito. La alternativa que encontramos fue utilizar un Dataset en Inglés (ya etiquetado) y traducirlo al Castellano.
   - **Nota:** La Traducción podría arruinar la Etiqueta. ¿Clasificador?

2. Buscar Modelos de Lenguaje que puedan trabajar en la tarea de Adaptación de textos en Castellano.
   - **Mistral** (aprobado), **Aya** (probando), **Qwen1.5** (probando).

3. Conseguir un Clasificador de Texto para los niveles MCER en Castellano.
   - No pudimos conseguir un Clasificador de Texto en Castellano. La alternativa fue traducir los textos Adaptados al inglés y utilizar un Clasificador en Inglés.

4. Para cada Modelo de Lenguaje, comparar precisión y velocidad de Adaptación de textos.

5. Fine-Tuning del Modelo de Lenguaje con mejor rendimiento.

## Técnicas relevantes para aplicar y justificación
- **Matriz de Confusión**.

## Referencias
- **Modelos de Lenguajes:**
  - [Mistral AI | Frontier AI in your hands](https://mistral.ai/)
  - [CohereForAI/aya-101 · Hugging Face](https://huggingface.co/CohereForAI/aya-101)
  - [Qwen1.5 - a Qwen Collection](https://huggingface.co/collections/Qwen/qwen15-65c0a2f577b1ecb76d786524)

- **Dataset en Inglés:** [CEFR Levelled English Texts](https://www.kaggle.com/datasets/amontgomerie/cefr-levelled-english-texts)
- **Traductor EN-ES:** [Helsinki-NLP/opus-mt-en-es](https://huggingface.co/Helsinki-NLP/opus-mt-en-es)
  - **Traducción (Código):** [notebook9834025409 | Kaggle](https://www.kaggle.com/code/alexistomascenteno/notebook9834025409/edit/run/197471934)
- **Traductor ES-EN:** [Helsinki-NLP/opus-mt-es-en](https://huggingface.co/Helsinki-NLP/opus-mt-es-en)
- **Clasificador de Texto:** [AbdulSami/bert-base-cased-cefr](https://huggingface.co/AbdulSami/bert-base-cased-cefr)
  - **Notebook de Mili:** [1.0 Using pretrained LLM for text classification - Colab](https://colab.research.google.com/drive/1h3hQ8anuKjoWJXz12p-OgwduBpYQB7rI?usp=sharing)

## Planificación
- **Semana 1:** Obtener recursos. Conseguir los Datasets, Modelos de Lenguajes y Clasificador de texto.
  - Objetivos 1, 2 y 3.

- **Semana 2 y 3:** Evaluación de precisión y velocidad de cada Modelo de Lenguaje. Decidir el mejor.
  - Objetivo 4.

- **Semana 4 y 5:** Fine-Tuning del mejor Modelo de Lenguaje. Comparaciones, conclusiones e informes.
  - Objetivo 5 e informe.