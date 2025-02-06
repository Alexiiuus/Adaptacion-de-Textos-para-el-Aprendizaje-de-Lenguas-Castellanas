Aquí tienes una versión mejorada de tu README:  

---

# Fine-Tuning del Modelo Cohere  

Este repositorio contiene el proceso de fine-tuning del modelo Cohere utilizando distintos conjuntos de datos preparados. El objetivo es analizar y comparar los resultados del entrenamiento bajo diferentes condiciones.  

## 📌 Conjuntos de datos  

Se han generado y subido a Cohere los siguientes datasets:  

- **`Sin filtros`**: Incluye el dataset original sin aplicar ningún filtro.  
- **`Exactos`**: Contiene solo los textos en los que el clasificador asigna el mismo nivel que el original.  
- **`Exactos y Adyacentes`**: Incluye los textos en los que el clasificador asigna el mismo nivel o un nivel adyacente al original.  
- **`Exactos + Mitad de Textos Adyacentes`**: Similar al dataset **Exactos y Adyacentes**, pero solo conserva la mitad de los textos adyacentes del dataset original.  

## 📌 Generación del Dataset Final  

Para generar el dataset final, cada texto seleccionado se incorpora en el siguiente *prompt*, el cual solicita una reformulación aleatoria del texto a un nivel específico, asegurando un balance en la cantidad de textos por nivel:  

```python
prompt = lambda label, text: f"""
A continuación, te proporcionaré un texto en español y te pediré que lo modifiques para diferentes niveles de competencia lingüística
(A1, A2, B1, B2, C1 y C2), concretamente: {label}. El objetivo es que adaptes el texto según el nivel de dificultad, modificando el
vocabulario y las estructuras gramaticales para que se ajusten a cada nivel, pero manteniendo el mismo mensaje central.  

Solo responde con la versión del texto modificada para dicho nivel. No incluyas ninguna introducción, título, explicación o comentario.  

Aquí está el texto:  
{text}
"""
```

## 📌 Requisitos  

Para entrenar modelos personalizados, es necesario:  

1. Crear una cuenta en [Cohere](https://mistral.ai/).  
2. Cargar los datasets en la plataforma de Cohere.  

## 📌 Contenido  

Este repositorio incluye dos **notebooks** con las pruebas realizadas:  

- 📌 **`FineTuned Pruebas.ipynb`**: Evalúa los distintos datasets probando el *prompt* en el modelo Cohere, tanto sin fine-tuning como con los modelos ajustados, obteniendo métricas de **Precisión Exacta** y **Precisión Aproximada**.  
- 📌 **`FineTuning Cohere.ipynb`**: Contiene el proceso de entrenamiento del modelo en cada uno de los datasets generados.  


