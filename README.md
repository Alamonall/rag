# Создание "очищенного" датасета на основе "Властелин коелец"

1. Заменяем термины в файлах и создаем новые файлы в `knowledge_base_new`
   1. ```cd replace_terms && npm ci && npx tsx index.ts```

# Создание векторного индекса базы знаний

## Какая модель использовалась

   Название модели: sentence-transformers/all-MiniLM-L6-v2
   Ссылку на репозиторий: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
   Размер эмбеддингов: 724 x 384
   Какая база знаний: FAISS
   Сколько чанков в индексе: 500
   Сколько времени заняла генерация: 45s

# Действия для генерации

```
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

1. Делим доки на части:
   1. ```bash python document-processor.py```
2. Создаем ембеддинги (векторы чанков)
   1. ```bash python generate_embeddings.py```
3.
