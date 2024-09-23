# News Trend Analyzer

Un script en Python que obtiene datos de una API pública de noticias, analiza la información y genera un informe estructurado.

## Objetivo

El objetivo de este proyecto es crear un script que:

1. Obtenga las noticias más recientes de Argentina a través de la API de NewsAPI.
2. Analice la información de los artículos, identifique palabras clave, categorice los artículos y genere un informe en formato JSON.

Nota: Para este fin solo se utilizaron bibliotecas estándar de Python o de terceros populares evitando entrar en el campo de librerías de Deep Learning más complejas.

## Requisitos

- Python 3.7+
- API Key de NewsAPI (https://newsapi.org/)

## Instalación y ejecución

1. Clonar el repositorio.
2. Crear un archivo `.env` en la raíz del proyecto y agregar la API key proporcionada nombrada como en el archivo `.env.example`
3. Instalar dependencias:

```bash
   pip install -r requirements.txt
```

4. Ejecutar

```bash
   python news_trend_analysis.py
```

## Observaciones

Dado que en las pruebas realizadas durante el desarrollo la consulta a la API utilizando el argumento 'ar' no arrojaba resultados se incorporó la función auxiliar get_google_news_ar() con el objetivo de obtener algunas noticias para la prueba de funcionamiento del script, asi como la posibilidad de ejecutar utilizando el argumento 'us' que obtiene mayor cantidad y variedad de noticias.

```bash
   python news_trend_analysis.py --country us
```

Para este último caso resulta menos eficaz la clasificación por categoría ya que se utiliza un dataset etiquetado en español (generado por ChatGPT) para la clasificación utilizando el método de Naive Bayes.
