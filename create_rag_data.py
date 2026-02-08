import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

from config import *

warnings.filterwarnings('ignore')

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================
DEFAULT_CONFIG = {
    "input_dir": INPUT_DIR,
    "output_dir": OUTPUT_DIR,
    "chunk_size": CHUNKS_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "embedding_model": EMBEDDING_MODEL,
    "faiss_index_type": "flat",  # flat, ivf, hnsw
    "min_chunk_length": 50,  # минимальная длина чанка в символах
    "max_chunk_length": 2000,  # максимальная длина чанка
    "device": "cpu",  # cpu, mps, cuda
}

# ============================================================================
# УТИЛИТЫ ДЛЯ РАБОТЫ С ФАЙЛАМИ
# ============================================================================

def find_text_files(directory: str) -> List[str]:
    """Найти все текстовые файлы в директории"""
    extensions = ['.md', '.markdown', '.txt', '.MD', '.Markdown', '.TXT']
    files = []
    
    for ext in extensions:
        files.extend(list(Path(directory).rglob(f"*{ext}")))
    
    return sorted([str(f) for f in files])

def read_file_with_encodings(filepath: str) -> str:
    """Чтение файла с автоматическим определением кодировки"""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1251', 'cp866', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # Если ни одна кодировка не подошла, пробуем с игнорированием ошибок
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"    ❌ Не удалось прочитать файл {filepath}: {e}")
        return ""

def clean_text(text: str) -> str:
    """Очистка текста"""
    if not text:
        return ""
    
    # Убираем лишние пробелы и переносы
    lines = []
    for line in text.split('\n'):
        line = line.rstrip()
        if line or (lines and lines[-1]):
            lines.append(line)
    
    return '\n'.join(lines)

# ============================================================================
# СОЗДАНИЕ ЧАНКОВ (ПРАВИЛЬНЫЙ ФОРМАТ)
# ============================================================================

class ChunkCreator:
    """Создатель чанков в правильном формате"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def split_by_paragraphs(self, text: str) -> List[str]:
        """Разделение текста на абзацы"""
        paragraphs = []
        current_paragraph = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(line)
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return paragraphs
    
    def split_long_paragraph(self, paragraph: str, chunk_size: int) -> List[str]:
        """Разделение длинного абзаца на чанки"""
        words = paragraph.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 для пробела
            
            if current_length + word_length > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Сохраняем overlap (последние N слов)
                overlap_words = max(1, int(len(current_chunk) * 0.3))  # 30% перекрытия
                current_chunk = current_chunk[-overlap_words:] if len(current_chunk) > overlap_words else current_chunk
                current_length = sum(len(w) + 1 for w in current_chunk) - 1
            
            current_chunk.append(word)
            current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def create_chunks_from_text(self, text: str, source: str) -> List[Dict]:
        """Создание чанков из текста"""
        chunks = []
        
        # Разделяем на абзацы
        paragraphs = self.split_by_paragraphs(text)
        
        for para_idx, paragraph in enumerate(paragraphs):
            para_length = len(paragraph)
            
            # Если абзац слишком короткий, пропускаем
            if para_length < self.config["min_chunk_length"]:
                continue
            
            # Если абзац слишком длинный, разбиваем его
            if para_length > self.config["chunk_size"]:
                para_chunks = self.split_long_paragraph(paragraph, self.config["chunk_size"])
                for chunk_idx, chunk_text in enumerate(para_chunks):
                    if len(chunk_text) < self.config["min_chunk_length"]:
                        continue
                    
                    chunks.append({
                        "text": chunk_text,
                        "source": source,
                        "paragraph": para_idx,
                        "chunk_in_paragraph": chunk_idx,
                        "char_count": len(chunk_text),
                        "word_count": len(chunk_text.split())
                    })
            else:
                # Абзац подходит как целый чанк
                chunks.append({
                    "text": paragraph,
                    "source": source,
                    "paragraph": para_idx,
                    "chunk_in_paragraph": 0,
                    "char_count": para_length,
                    "word_count": len(paragraph.split())
                })
        
        return chunks
    
    def process_files(self) -> List[Dict]:
        """Обработка всех файлов и создание чанков"""
        print("🔧 Создание чанков...")
        
        input_dir = Path(self.config["input_dir"])
        if not input_dir.exists():
            print(f"❌ Входная директория не существует: {input_dir}")
            return []
        
        # Находим файлы
        files = find_text_files(str(input_dir))
        if not files:
            print(f"❌ Не найдено файлов в {input_dir}")
            print("   Поддерживаемые расширения: .md, .markdown, .txt")
            return []
        
        print(f"📁 Найдено {len(files)} файлов")
        
        all_chunks = []
        for file_idx, filepath in enumerate(files, 1):
            filename = Path(filepath).name
            print(f"  [{file_idx}/{len(files)}] 📄 {filename}")
            
            # Чтение файла
            content = read_file_with_encodings(filepath)
            if not content.strip():
                print(f"     ⚠️  Файл пуст")
                continue
            
            # Очистка текста
            cleaned_content = clean_text(content)
            
            # Создание чанков
            file_chunks = self.create_chunks_from_text(cleaned_content, filepath)
            
            print(f"     ✅ Создано {len(file_chunks)} чанков")
            all_chunks.extend(file_chunks)
        
        # Добавляем ID к чанкам
        for idx, chunk in enumerate(all_chunks):
            chunk["id"] = idx
        
        print(f"\n🎉 Всего создано {len(all_chunks)} чанков")
        return all_chunks
    
    def save_metadata(self, chunks: List[Dict], output_path: str):
        """Сохранение метаданных в ПРАВИЛЬНОМ формате"""
        print(f"\n💾 Сохранение metadata.json...")
        
        # ПРАВИЛЬНЫЙ ФОРМАТ: просто список чанков
        metadata = chunks  # Уже правильный формат
        
        # Создаем директорию если не существует
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ metadata.json сохранен: {len(metadata)} записей")
        
        # Сохраняем статистику отдельно
        stats = self.calculate_statistics(chunks)
        stats_path = Path(output_path).parent / "statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"📊 statistics.json сохранен")
        
        # Создаем превью
        self.create_preview(chunks, Path(output_path).parent / "chunks_preview.txt")
    
    def calculate_statistics(self, chunks: List[Dict]) -> Dict:
        """Расчет статистики по чанкам"""
        if not chunks:
            return {}
        
        char_counts = [chunk["char_count"] for chunk in chunks]
        word_counts = [chunk["word_count"] for chunk in chunks]
        
        # Группируем по файлам
        files = {}
        for chunk in chunks:
            source = chunk["source"]
            if source not in files:
                files[source] = {
                    "filename": Path(source).name,
                    "chunks": 0,
                    "total_chars": 0,
                    "total_words": 0
                }
            files[source]["chunks"] += 1
            files[source]["total_chars"] += chunk["char_count"]
            files[source]["total_words"] += chunk["word_count"]
        
        return {
            "total_chunks": len(chunks),
            "total_files": len(files),
            "avg_chars_per_chunk": sum(char_counts) / len(char_counts),
            "min_chars_per_chunk": min(char_counts),
            "max_chars_per_chunk": max(char_counts),
            "avg_words_per_chunk": sum(word_counts) / len(word_counts),
            "min_words_per_chunk": min(word_counts),
            "max_words_per_chunk": max(word_counts),
            "files": files
        }
    
    def create_preview(self, chunks: List[Dict], preview_path: str):
        """Создание превью чанков"""
        print(f"👀 Создание превью...")
        
        with open(preview_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ПРЕВЬЮ СОЗДАННЫХ ЧАНКОВ\n")
            f.write("=" * 80 + "\n\n")
            
            for i, chunk in enumerate(chunks[:10]):  # Первые 10 чанков
                f.write(f"ЧАНК {i+1} (ID: {chunk['id']})\n")
                f.write(f"Файл: {Path(chunk['source']).name}\n")
                f.write(f"Абзац: {chunk['paragraph']}, Чанк: {chunk['chunk_in_paragraph']}\n")
                f.write(f"Символов: {chunk['char_count']}, Слов: {chunk['word_count']}\n")
                f.write("-" * 40 + "\n")
                f.write(chunk['text'][:300] + ("..." if len(chunk['text']) > 300 else "") + "\n")
                f.write("=" * 80 + "\n\n")

# ============================================================================
# СОЗДАНИЕ ЭМБЕДДИНГОВ
# ============================================================================

class EmbeddingCreator:
    """Создатель эмбеддингов"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def create_embeddings(self, chunks: List[Dict]) -> Optional[np.ndarray]:
        """Создание эмбеддингов для чанков"""
        print("\n🧠 Создание эмбеддингов...")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Загружаем модель
            print(f"  Загрузка модели: {self.config['embedding_model']}")
            model = SentenceTransformer(
                self.config['embedding_model'],
                device=self.config['device']
            )
            
            # Извлекаем тексты
            texts = [chunk["text"] for chunk in chunks]
            print(f"  Обработка {len(texts)} текстов...")
            
            # Создаем эмбеддинги
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True  # Для косинусной схожести
            )
            
            print(f"✅ Эмбеддинги созданы: {embeddings.shape}")
            return embeddings
            
        except ImportError:
            print("❌ sentence-transformers не установлен")
            print("   Установите: pip install sentence-transformers")
            return None
        except Exception as e:
            print(f"❌ Ошибка создания эмбеддингов: {e}")
            return None
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: str):
        """Сохранение эмбеддингов"""
        print(f"\n💾 Сохранение эмбеддингов...")
        
        # Создаем директорию если не существует
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path, embeddings)
        
        # Сохраняем информацию об эмбеддингах
        info = {
            "embedding_model": self.config["embedding_model"],
            "embedding_dim": embeddings.shape[1],
            "num_embeddings": embeddings.shape[0],
            "normalized": True
        }
        
        info_path = Path(output_path).parent / "embeddings_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
        
        print(f"✅ Эмбеддинги сохранены: {output_path}")
        print(f"   Размер: {embeddings.shape[0]} x {embeddings.shape[1]}")

# ============================================================================
# СОЗДАНИЕ FAISS ИНДЕКСА
# ============================================================================

class FaissIndexCreator:
    """Создатель FAISS индекса"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def create_index(self, embeddings: np.ndarray) -> Any:
        """Создание FAISS индекса"""
        print("\n🔍 Создание FAISS индекса...")
        
        try:
            import faiss
            
            dimension = embeddings.shape[1]
            print(f"  Размерность: {dimension}")
            print(f"  Количество векторов: {embeddings.shape[0]}")
            
            # Выбираем тип индекса
            if self.config["faiss_index_type"] == "flat":
                # Точный поиск
                index = faiss.IndexFlatIP(dimension)  # Inner Product для косинусной
                print("  Тип: FlatIP (точный поиск)")
                
            elif self.config["faiss_index_type"] == "ivf":
                # Приближенный поиск (быстрее)
                nlist = min(100, embeddings.shape[0] // 39)
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                
                # Требуется обучение
                print(f"  Обучение IVF индекса (nlist={nlist})...")
                index.train(embeddings)
                index.nprobe = 10
                print("  Тип: IVF (приближенный поиск)")
                
            elif self.config["faiss_index_type"] == "hnsw":
                # HNSW (иерархический навигационный малый мир)
                index = faiss.IndexHNSWFlat(dimension, 32)
                index.hnsw.efConstruction = 200
                index.hnsw.efSearch = 128
                print("  Тип: HNSW (быстрый приближенный поиск)")
                
            else:
                print(f"❌ Неизвестный тип индекса: {self.config['faiss_index_type']}")
                return None
            
            # Добавляем векторы в индекс
            print("  Добавление векторов...")
            index.add(embeddings)
            
            print(f"✅ Индекс создан: {index.ntotal} векторов")
            return index
            
        except ImportError:
            print("❌ faiss не установлен")
            print("   Установите: pip install faiss-cpu (или faiss-gpu для CUDA)")
            return None
        except Exception as e:
            print(f"❌ Ошибка создания индекса: {e}")
            return None
    
    def save_index(self, index: Any, output_path: str):
        """Сохранение FAISS индекса"""
        print(f"\n💾 Сохранение FAISS индекса...")
        
        try:
            import faiss
            
            # Создаем директорию если не существует
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            faiss.write_index(index, output_path)
            
            # Сохраняем информацию об индексе
            info = {
                "index_type": self.config["faiss_index_type"],
                "num_vectors": index.ntotal,
                "dimension": index.d,
                "faiss_version": faiss.__version__
            }
            
            info_path = Path(output_path).parent / "faiss_index_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2)
            
            size_mb = Path(output_path).stat().st_size / 1024 / 1024
            print(f"✅ Индекс сохранен: {output_path}")
            print(f"   Размер: {size_mb:.2f} MB")
            
        except Exception as e:
            print(f"❌ Ошибка сохранения индекса: {e}")
    
    def test_index(self, index: Any, embeddings: np.ndarray, num_tests: int = 3):
        """Тестирование индекса"""
        print("\n🧪 Тестирование индекса...")
        
        try:
            import faiss
            
            for i in range(min(num_tests, embeddings.shape[0])):
                # Берем i-й вектор как тестовый запрос
                query = embeddings[i:i+1]
                
                # Ищем 3 ближайших соседа
                distances, indices = index.search(query, k=3)
                
                print(f"  Тест {i+1}:")
                print(f"    Найденные индексы: {indices[0].tolist()}")
                print(f"    Расстояния: {distances[0].round(4).tolist()}")
                
                # Проверяем, что первый результат - сам вектор
                if indices[0][0] == i:
                    print(f"    ✅ Вектор нашел себя (расстояние: {distances[0][0]:.4f})")
                else:
                    print(f"    ⚠️  Вектор не нашел себя на первом месте")
            
            print("✅ Тестирование завершено")
            
        except Exception as e:
            print(f"❌ Ошибка тестирования: {e}")

# ============================================================================
# ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================================

class RAGDataPipeline:
    """Полный пайплайн создания данных для RAG"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Создаем выходную директорию
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Пути к файлам
        self.metadata_path = self.output_dir / "metadata.json"
        self.embeddings_path = self.output_dir / "embeddings.npy"
        self.faiss_index_path = self.output_dir / "faiss.index"
        
    def run(self, test_index: bool = True):
        """Запуск полного пайплайна"""
        print("🚀 ЗАПУСК RAG DATA PIPELINE")
        print("=" * 60)
        
        # 1. Создание чанков
        creator = ChunkCreator(self.config)
        chunks = creator.process_files()
        
        if not chunks:
            print("❌ Не создано ни одного чанка")
            return False
        
        # 2. Сохранение метаданных
        creator.save_metadata(chunks, str(self.metadata_path))
        
        # 3. Создание эмбеддингов
        embedding_creator = EmbeddingCreator(self.config)
        embeddings = embedding_creator.create_embeddings(chunks)
        
        if embeddings is None:
            print("❌ Не удалось создать эмбеддинги")
            return False
        
        # 4. Сохранение эмбеддингов
        embedding_creator.save_embeddings(embeddings, str(self.embeddings_path))
        
        # 5. Создание FAISS индекса
        index_creator = FaissIndexCreator(self.config)
        index = index_creator.create_index(embeddings)
        
        if index is None:
            print("❌ Не удалось создать индекс")
            return False
        
        # 6. Сохранение индекса
        index_creator.save_index(index, str(self.faiss_index_path))
        
        # 7. Тестирование индекса (опционально)
        if test_index:
            index_creator.test_index(index, embeddings)
        
        print("\n" + "=" * 60)
        print("🎉 ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
        print("=" * 60)
        
        # Показываем созданные файлы
        self.show_results()
        
        return True
    
    def show_results(self):
        """Показать результаты работы пайплайна"""
        print("\n📁 СОЗДАННЫЕ ФАЙЛЫ:")
        print(f"  📄 Метаданные: {self.metadata_path}")
        print(f"  🧠 Эмбеддинги: {self.embeddings_path}")
        print(f"  🔍 FAISS индекс: {self.faiss_index_path}")
        print(f"  📊 Статистика: {self.output_dir / 'statistics.json'}")
        print(f"  👀 Превью: {self.output_dir / 'chunks_preview.txt'}")
        print(f"  ℹ️  Инфо об эмбеддингах: {self.output_dir / 'embeddings_info.json'}")
        print(f"  ℹ️  Инфо об индексе: {self.output_dir / 'faiss_index_info.json'}")
        
        print("\n🤖 ДЛЯ ЗАПУСКА RAG БОТА:")
        print(f"  python rag_bot_eng.py \\")
        print(f"    --model ./mistral-7b-instruct-v0.2.Q4_K_M.gguf \\")
        print(f"    --index {self.faiss_index_path} \\")
        print(f"    --metadata {self.metadata_path}")
        
        print("\n💡 БЫСТРЫЙ ТЕСТ:")
        print(f'  python rag_bot_eng.py \\')
        print(f'    --model ./mistral-7b-instruct-v0.2.Q4_K_M.gguf \\')
        print(f'    --index {self.faiss_index_path} \\')
        print(f'    --metadata {self.metadata_path} \\')
        print(f'    --question "What is machine learning?"')

# ============================================================================
# КОМАНДНАЯ СТРОКА
# ============================================================================

def main():
    """Основная функция"""
    
    parser = argparse.ArgumentParser(
        description="Полный пайплайн создания данных для RAG системы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                         # Все параметры по умолчанию
  %(prog)s --input ./my_docs      # Указать свою папку с документами
  %(prog)s --chunk-size 800       # Увеличить размер чанков
  %(prog)s --embed-model sentence-transformers/all-mpnet-base-v2  # Другая модель
  %(prog)s --index-type ivf       # Быстрый поиск для больших коллекций
  
Результат:
  Все файлы сохраняются в папку ./rag_data/
  metadata.json - в правильном формате (простой список)
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        default=INPUT_DIR,
        help="Входная директория с документами"
    )
    
    parser.add_argument(
        "--output", "-o",
        default=OUTPUT_DIR,
        help="Выходная директория для всех файлов"
    )
    
    parser.add_argument(
        "--chunk-size", "-s",
        type=int,
        default=CHUNKS_SIZE,
        help="Максимальный размер чанка в символах"
    )
    
    parser.add_argument(
        "--chunk-overlap", "-l",
        type=int,
        default=CHUNK_OVERLAP,
        help="Перекрытие между чанками"
    )
    
    parser.add_argument(
        "--embed-model", "-e",
        default=EMBEDDING_MODEL,
        help="Модель для создания эмбеддингов"
    )
    
    parser.add_argument(
        "--index-type", "-t",
        default="flat",
        choices=["flat", "ivf", "hnsw"],
        help="Тип FAISS индекса"
    )
    
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Устройство для вычислений (mps для Mac M1/M2/M3)"
    )
    
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Не тестировать индекс после создания"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Установить зависимости и выйти"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Быстрый режим (меньше чанков для тестирования)"
    )
    
    args = parser.parse_args()
    
    # Установка зависимостей
    if args.install_deps:
        print("📦 Установка зависимостей...")
        
        deps = [
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
            "numpy>=1.24.0",
        ]
        
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + deps)
            print("✅ Зависимости установлены!")
            print("\nТеперь можно запустить пайплайн:")
            print("  python rag_create_data.py")
        except subprocess.CalledProcessError as e:
            print(f"❌ Ошибка установки: {e}")
            print("\nУстановите вручную:")
            print("pip install sentence-transformers faiss-cpu numpy")
        return
    
    # Конфигурация
    config = DEFAULT_CONFIG.copy()
    config.update({
        "input_dir": args.input,
        "output_dir": args.output,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "embedding_model": args.embed_model,
        "faiss_index_type": args.index_type,
        "device": args.device,
    })
    
    # Быстрый режим
    if args.quick:
        config["chunk_size"] = 300
        config["chunk_overlap"] = 30
        print("⚡ Быстрый режим активирован")
    
    try:
        # Запуск пайплайна
        pipeline = RAGDataPipeline(config)
        success = pipeline.run(test_index=not args.no_test)
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    # Проверка Python версии
    if sys.version_info < (3, 7):
        print("❌ Требуется Python 3.7 или выше")
        sys.exit(1)
    
    # Запуск
    main()
