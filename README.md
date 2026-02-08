# Последовательность действий

1. Заменяем термины в файлах и создаем новые файлы в `knowledge_base_new`
   1. ```cd replace_terms && npm ci && npx tsx index.ts```
2. Делим доки на части:
   1. ```bash python document-processor.py```
3. Создаем ембеддинги (векторы чанков)
   1. ```bash python generate_embeddings.py```
4.
