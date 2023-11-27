# NLP UA Locations Extractions

First place solution of the local hakathon (10 participating teams), targeted for creating a model for automatic locations extraction from the Telegram posts in Ukrainian and russian languages, facilating further geolocation of events.

## Deployed model

Url: [https://ua-locations-b257fnr7mq-uc.a.run.app](https://ua-locations-b257fnr7mq-uc.a.run.app) 

An optimized ONNX version of the model is wrapped with simple REST API and deployed to the Google Cloud Platform. Comparison of speed and performance of model compressions can be found at [notebooks/quantization&optimization](notebooks/quantization&optimization.ipynb). The API scheme can be found at the url above. Below is a sample request with python to the api's endpoint.

```Python
import requests
import json

texts = [
"""Сьогодні у Києві повітря більш забруднене, ніж зазвичай \n\nНа Харківському масиві шостий день поспіль спостерігається перевищення гранично допустимої концентрації сірководню. \n\nНа лівому березі моніторинг на Харківському шосе показав індекс 55, а на вулиці Архітектора Вербицького – 52.  \n\nНа правому березі пункт моніторингу на вулиці Турівській зафіксував позначку – 34, на Щусєва – 33, на проспекті Правди – 31, а на вулиці Китаївській – 37. """,
"""Житомирська область, до уваги водіїв ❗️\n\nУ зв’язку з дорожньо-транспортною пригодою з потерпілими, що сталася, близько 18-ї години, на 132 км автодороги М-06 Київ — Чоп (поблизу села Вереси Житомирської області), рух транспорту частково ускладнений в обох напрямках.\n\nПлануйте маршрут своєї поїздки з урахуванням цієї інформації. \n\nПідписатися | Запропонувати новину | Реклама | Автострахування | Страхування життя """
]

url = "https://ua-locations-b257fnr7mq-uc.a.run.app/api/extract_locations/"

response = requests.post(url, data=json.dumps({"texts": test_texts}))

print(json.dumps(json.loads(response.text), indent=4, ensure_ascii=False))
```

```
# Result 
[
    [
        "Києві",
        "Харківському масиві",
        "Харківському шосе",
        "вулиці Архітектора Вербицького",
        "вулиці Турівській",
        "Щусєва – 33",
        "проспекті Правди – 31",
        "вулиці Китаївській"
    ],
    [
        "М-06",
        "Чоп",
        "Вереси",
        "Житомирської області"
    ]
]
```

## Build container


