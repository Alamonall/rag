#!/usr/bin/env python3
"""
English RAG Bot - совместимый с форматом из rag_create_data.py
"""

import os
import sys
import json
import argparse
import readline
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import requests

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

# ============================================================================
# CONFIGURATION
# ============================================================================
from config import * 

@dataclass
class Config:
    """Bot configuration"""
    model_path: str = RAG_MODEL
    faiss_index_path: str = FAISS_INDEX
    metadata_path: str = METADATA_FILE
    embed_model: str = EMBEDDING_MODEL
    device: str = "mps"  # mps for Mac, cpu for others
    num_sources: int = 3
    stream_responses: bool = True
    context_window: int = 4096
    max_tokens: int = 512
    temperature: float = 0.1

# ============================================================================
# DATA MODELS (совместимые с новым форматом)
# ============================================================================

@dataclass
class Chunk:
    """Text chunk - совместим с форматом из rag_create_data.py"""
    id: int
    text: str
    source: str
    # Новые поля из rag_create_data.py
    paragraph: int = 0
    chunk_in_paragraph: int = 0
    char_count: int = 0
    word_count: int = 0
    # Старое поле для совместимости
    position: int = 0  # Используем paragraph как position для совместимости
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Убедимся, что position есть (для совместимости)
        if self.position == 0 and self.paragraph != 0:
            self.position = self.paragraph

@dataclass
class Document:
    """Document with chunks"""
    source: str
    chunks: List[Chunk]
    
    @property
    def title(self) -> str:
        """Document title from filename"""
        return Path(self.source).stem.replace('_', ' ').replace('-', ' ').title()

# ============================================================================
# UTILITES
# ============================================================================

def ensure_model_exists(model_path: str, model_url: str = None) -> str:
    """
    Проверяет наличие модели и загружает если нужно
    
    Args:
        model_path: Локальный путь к модели
        model_url: URL для скачивания (если None, используется стандартный)
    
    Returns:
        Путь к локальному файлу модели
    """
    path = Path(model_path)
    
    # Если файл существует
    if path.exists():
        # Проверяем размер файла (минимальная проверка)
        file_size = path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # Минимум 100MB для модели
            print(f"Модель найдена: {model_path} ({file_size / 1024 / 1024:.1f} MB)")
            return model_path
        else:
            print(f"Файл слишком маленький ({file_size} байт), перезагружаем...")
            path.unlink()
    
    # Если URL не передан, используем стандартный
    if model_url is None:
        model_url = RAG_URL
    
    # Создаем директорию если нужно
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Загружаем модель
    return download_model(model_url, model_path)

def download_model(model_url: str, model_path: str = None):
    """Скачивает модель по ссылке"""
    # Преобразуем в Path
    if model_path is None:
        model_path = Path(model_url.split("/")[-1])
    else:
        model_path = Path(model_path)
    
    # Создаем директорию
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Проверяем существует ли файл
    if model_path.exists():
        print(f"Модель уже существует: {model_path}")
        return str(model_path)
    
    print(f"Скачивание модели из {model_url}...")
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        # Получаем размер файла для прогресса
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                
                # Простой прогресс-бар
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rПрогресс: {percent:.1f}%", end="")
        
        print(f"\nМодель сохранена в {model_path}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        if model_path.exists():
            model_path.unlink()
        raise
    
    return str(model_path)

# ============================================================================
# MAIN RAG BOT CLASS (FIXED FOR NEW FORMAT)
# ============================================================================

class RAGCliBot:
    def __init__(self, config: Config):
        """
        Initialize English RAG Bot - совместимый с новым форматом
        """
        self.config = config
        self._check_files()
        
        print("🔧 Initializing English RAG Bot...")
        print(f"  Model: {Path(config.model_path).name}")
        print(f"  Index: {config.faiss_index_path}")
        print(f"  Metadata: {config.metadata_path}")
        print(f"  Embeddings: {config.embed_model}")
        
        # Load metadata with new format support
        print("📖 Loading metadata...")
        self.chunks, self.documents = self._load_metadata_new_format()
        
        print("🔍 Loading FAISS index...")
        self.index = faiss.read_index(config.faiss_index_path)
        
        # Validate consistency
        if len(self.chunks) != self.index.ntotal:
            print(f"⚠️  Warning: chunks={len(self.chunks)}, vectors={self.index.ntotal}")
            if len(self.chunks) > self.index.ntotal:
                print(f"   Using first {self.index.ntotal} chunks")
                self.chunks = self.chunks[:self.index.ntotal]
        
        # Load models
        print(f"📊 Loading embedding model: {config.embed_model}")
        self.embedder = SentenceTransformer(config.embed_model, device=config.device)
        
        print(f"🧠 Loading LLM: {Path(config.model_path).name}")
        self.llm = self._init_llm()
        
        print(f"✅ Ready! Documents: {len(self.documents)}, Chunks: {len(self.chunks)}")
        print("-" * 50)
    
    def _check_files(self):
        """Check if required files exist"""
        missing = []
        for path, name in [
            (self.config.model_path, "LLM Model"),
            (self.config.faiss_index_path, "FAISS Index"),
            (self.config.metadata_path, "Metadata")
        ]:
            if not Path(path).exists():
                missing.append(f"{name}: {path}")
        
        if missing:
            print("❌ Missing files:")
            for msg in missing:
                print(f"  {msg}")
            sys.exit(1)
    
    def _load_metadata_new_format(self):
        """
        Load metadata in new format from rag_create_data.py
        Формат: список словарей с полями:
        - id, text, source, paragraph, chunk_in_paragraph, char_count, word_count
        """
        try:
            with open(self.config.metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"  Metadata type: {type(data).__name__}")
            
            if not isinstance(data, list):
                print(f"❌ Metadata should be a list, got {type(data)}")
                sys.exit(1)
            
            print(f"  Found {len(data)} chunks")
            
            # Process chunks
            chunks = []
            docs_dict = defaultdict(list)
            
            for item in data:
                if not isinstance(item, dict):
                    print(f"⚠️  Skipping non-dict item: {type(item)}")
                    continue
                
                # Создаем чанк из нового формата
                chunk = Chunk(
                    id=item.get('id', len(chunks)),
                    text=item.get('text', ''),
                    source=item.get('source', 'unknown'),
                    paragraph=item.get('paragraph', 0),
                    chunk_in_paragraph=item.get('chunk_in_paragraph', 0),
                    char_count=item.get('char_count', 0),
                    word_count=item.get('word_count', 0),
                    position=item.get('position', item.get('paragraph', 0)),  # Для совместимости
                    metadata={
                        'char_count': item.get('char_count', 0),
                        'word_count': item.get('word_count', 0),
                        'paragraph': item.get('paragraph', 0),
                        'chunk_in_paragraph': item.get('chunk_in_paragraph', 0),
                        'original_data': item  # Сохраняем все оригинальные данные
                    }
                )
                
                if chunk.text.strip():  # Пропускаем пустые чанки
                    chunks.append(chunk)
                    docs_dict[chunk.source].append(chunk)
            
            # Create documents
            documents = {}
            for source, source_chunks in docs_dict.items():
                # Сортируем по paragraph и chunk_in_paragraph
                source_chunks.sort(key=lambda x: (x.paragraph, x.chunk_in_paragraph))
                documents[source] = Document(source=source, chunks=source_chunks)
            
            print(f"  Processed {len(chunks)} chunks from {len(documents)} documents")
            return chunks, documents
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in metadata.json: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading metadata: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _init_llm(self):
        """Initialize LLM model"""
        return Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.context_window,
            n_gpu_layers=20 if self.config.device == "mps" else 0,
            n_threads=8,
            n_batch=512,
            use_mmap=True,
            use_mlock=True,
            verbose=False,
            seed=42,
        )
    
    def search(self, query: str, k: int = None):
        """Search for similar chunks"""
        if k is None:
            k = self.config.num_sources
        
        # Embed query
        query_embedding = self.embedder.encode([query])
        
        # Search in FAISS
        distances, indices = self.index.search(query_embedding, k)
        
        # Build results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                doc = self.documents.get(chunk.source)
                
                if doc:
                    results.append({
                        "chunk": chunk,
                        "document": doc,
                        "score": float(1 - dist),
                        "distance": float(dist),
                        "index": idx
                    })
        
        # Sort by relevance
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    def format_context(self, results):
        """Format context for prompt"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            chunk = result["chunk"]
            doc = result["document"]
            score = result["score"]
            
            # Форматируем источник
            source_info = f"{doc.title}"
            if hasattr(chunk, 'paragraph'):
                source_info += f" (paragraph {chunk.paragraph}"
                if hasattr(chunk, 'chunk_in_paragraph'):
                    source_info += f".{chunk.chunk_in_paragraph}"
                source_info += ")"
            
            context_parts.append(
                f"[Source {i}: {source_info}, relevance: {score:.1%}]\n"
                f"{chunk.text}\n"
            )
        
        return "\n".join(context_parts)
    
    def create_prompt(self, question: str, context: str) -> str:
            return f"""
System: You are a helper who thinks first and then responds. Always write down your steps.

RULES:
- Answer the question using ONLY the provided context.
- Use step-by-step reasoning internally to derive the answer.
- DO NOT reveal your chain-of-thought
- DO NOT reveal examples.
- If the answer is not present in the context, reply exactly: "I don't know."
- Follow the style of the examples.
- Don't make up an answer.

====================
CONTEXT:
{context}
====================

EXAMPLES:
Answer questions clearly and concisely, using the same style as in the examples below.

Example 1:
Question: What is Barok?
Answer: Barok is an Elven refuge in Yorkville, also known as Forgottentome. It is ruled by Marig and serves as a place of rest, healing, and counsel.

Example 2:
Question: Who is Taeth?
Answer: Taeth is a wizard sent to Yorkville to help oppose Sapphirefang. He is a member of the Yorkjaukstone and plays a key role in guiding the Free Peoples.

====================

Now answer the following question.

Question: {question}
Answer:"""
    
    def ask(self, question: str, k: int = None):
        """Ask a question and get answer"""
        if k is None:
            k = self.config.num_sources
        
        print(f"\n📝 Question: {question}")
        
        # Search for relevant chunks
        results = self.search(question, k=k)
        
        if not results:
            print("❌ No relevant information found")
            return {
                "question": question,
                "answer": "No information found in documents",
                "sources": []
            }
        
        # Show found sources
        print(f"🔍 Found {len(results)} sources:")
        for i, result in enumerate(results, 1):
            chunk = result["chunk"]
            doc = result["document"]
            score = result["score"]
            
            source_info = f"{doc.title}"
            if hasattr(chunk, 'paragraph'):
                source_info += f" (p{chunk.paragraph}"
                if hasattr(chunk, 'chunk_in_paragraph'):
                    source_info += f".{chunk.chunk_in_paragraph}"
                source_info += ")"
            
            preview = chunk.text[:80].replace('\n', ' ')
            print(f"{i}. {source_info} ({score:.1%})")
            print(f" preview:  {preview}...")
        
        print("-" * 50)
        
        # Create context and prompt
        context = self.format_context(results)
        prompt = self.create_prompt(question, context)
        
        # Generate response
        print("🤖 Answer: ", end="", flush=True)
        
        full_response = ""
        if self.config.stream_responses:
            # Streaming response
            stream = self.llm(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stream=True,
                echo=False
            )
            
            for output in stream:
                chunk = output['choices'][0]['text']
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print()
        else:
            # Direct response
            response = self.llm(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stream=False,
                echo=False
            )
            full_response = response['choices'][0]['text'].strip()
            print(full_response)
        
        # Return structured result
        return {
            "question": question,
            "answer": full_response,
            "sources": [
                {
                    "source": r["document"].title,
                    "filename": Path(r["chunk"].source).name,
                    "score": r["score"],
                    "paragraph": getattr(r["chunk"], 'paragraph', 0),
                    "chunk_in_paragraph": getattr(r["chunk"], 'chunk_in_paragraph', 0),
                    "position": r["chunk"].position,
                    "text_preview": r["chunk"].text[:100]
                }
                for r in results
            ]
        }
    
    def show_stats(self):
        """Show statistics about loaded data"""
        print("\n📊 STATISTICS:")
        print(f"  • Documents: {len(self.documents)}")
        print(f"  • Chunks: {len(self.chunks)}")
        print(f"  • FAISS index size: {self.index.ntotal} vectors")
        
        # Show document info
        print(f"\n📚 DOCUMENTS:")
        for i, (source, doc) in enumerate(list(self.documents.items())[:5], 1):
            print(f"  {i}. {doc.title}")
            print(f"     Chunks: {len(doc.chunks)}")
            print(f"     File: {Path(source).name}")
        
        if len(self.documents) > 5:
            print(f"     ... and {len(self.documents) - 5} more documents")
    
    def interactive_mode(self):
        """Interactive mode"""
        print("\n" + "="*60)
        print("🤖 ENGLISH RAG BOT - Interactive Mode")
        print("="*60)
        print(f"Model: {Path(self.config.model_path).name}")
        print(f"Documents: {len(self.documents)}")
        print(f"Chunks: {len(self.chunks)}")
        print(f"Sources to retrieve: {self.config.num_sources}")
        print("="*60)
        print("Commands:")
        print("  /exit, /quit  - Exit")
        print("  /stats        - Show statistics")
        print("  /sources N    - Use N sources (default: 3)")
        print("  /stream       - Toggle streaming")
        print("  /help         - Show help")
        print("="*60)
        
        current_k = self.config.num_sources
        
        while True:
            try:
                user_input = input("\n💭 > ").strip()
                
                if not user_input:
                    continue
                
                # Process commands
                if user_input.lower() in ['/exit', '/quit', 'exit', 'quit']:
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == '/stats':
                    self.show_stats()
                    continue
                
                elif user_input.lower().startswith('/sources'):
                    parts = user_input.split()
                    if len(parts) == 2 and parts[1].isdigit():
                        current_k = int(parts[1])
                        if 1 <= current_k <= 10:
                            print(f"✅ Now using {current_k} sources")
                        else:
                            print("❌ Please use 1-10 sources")
                    else:
                        print(f"ℹ️  Currently using {current_k} sources")
                    continue
                
                elif user_input.lower() == '/stream':
                    self.config.stream_responses = not self.config.stream_responses
                    status = "enabled" if self.config.stream_responses else "disabled"
                    print(f"✅ Streaming {status}")
                    continue
                
                elif user_input.lower() == '/help':
                    print("Just type your question in English.")
                    print(f"Currently using {current_k} sources")
                    print(f"Streaming: {'on' if self.config.stream_responses else 'off'}")
                    continue
                
                # Regular question
                result = self.ask(user_input, k=current_k)
                
                # # Optional: save to file
                # if len(user_input) < 50:  # Short questions only
                #     safe_name = "".join(c for c in user_input[:20] if c.isalnum())
                #     filename = f"rag_answer_{safe_name}.json"
                #     with open(filename, 'w', encoding='utf-8') as f:
                #         json.dump(result, f, ensure_ascii=False, indent=2)
                #     print(f"💾 Result saved to: {filename}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RAG Cli Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Interactive mode
  %(prog)s --question "What is AI?"  # Single question
  %(prog)s --stats                  # Show statistics
  
Required files from rag_create_data.py:
  • metadata.json in ./rag_data/ (new format)
  • faiss.index in ./rag_data/
  • LLM model in GGUF format
        """
    )
    
    parser.add_argument(
        "--model", "-m",
        default=RAG_MODEL,
        help="Path to GGUF LLM model"
    )
    
    parser.add_argument(
        "--index", "-i",
        default=FAISS_INDEX,
        help="Path to FAISS index"
    )
    
    parser.add_argument(
        "--metadata", "-d",
        default=METADATA_FILE,
        help="Path to metadata.json"
    )
    
    parser.add_argument(
        "--embed-model", "-e",
        default=EMBEDDING_MODEL,
        help="Embedding model"
    )
    
    parser.add_argument(
        "--question", "-q",
        help="Ask a single question"
    )
    
    parser.add_argument(
        "--sources", "-k",
        type=int,
        default=3,
        help="Number of sources to retrieve"
    )
    
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable response streaming"
    )
    
    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cpu"],
        help="Device for computations"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics and exit"
    )
    
    parser.add_argument(
        "--check-format",
        action="store_true",
        help="Check metadata format and exit"
    )
    
    args = parser.parse_args()
    
    # Check metadata format if requested
    if args.check_format:
        print("🔍 Checking metadata format...")
        try:
            with open(args.metadata, 'r') as f:
                data = json.load(f)
            
            print(f"  Type: {type(data).__name__}")
            
            if isinstance(data, list):
                print(f"  ✅ Correct format: list")
                print(f"  Length: {len(data)}")
                
                if len(data) > 0:
                    first = data[0]
                    print(f"  First item type: {type(first).__name__}")
                    
                    if isinstance(first, dict):
                        print(f"  Keys: {list(first.keys())}")
                        
                        # Check required fields
                        required = ['text', 'source']
                        missing = [f for f in required if f not in first]
                        
                        if missing:
                            print(f"  ❌ Missing fields: {missing}")
                        else:
                            print(f"  ✅ Required fields present")
                            
                            # Show example
                            print(f"\n  📋 Example:")
                            print(f"    ID: {first.get('id', 'N/A')}")
                            print(f"    Source: {first.get('source')}")
                            print(f"    Text: {first.get('text', '')[:50]}...")
                            print(f"    Paragraph: {first.get('paragraph', 'N/A')}")
                            print(f"    Char count: {first.get('char_count', 'N/A')}")
                    else:
                        print(f"  ❌ Items should be dictionaries")
            else:
                print(f"  ❌ Should be a list, got {type(data)}")
            
            sys.exit(0)
            
        except Exception as e:
            print(f"❌ Error checking format: {e}")
            sys.exit(1)
    
    # Create configuration
    config = Config(
        model_path=args.model,
        faiss_index_path=args.index,
        metadata_path=args.metadata,
        embed_model=args.embed_model,
        device=args.device,
        num_sources=args.sources,
        stream_responses=not args.no_stream
    )

    ensure_model_exists(config.model_path, RAG_URL)
    
    try:
        # Initialize bot
        bot = RAGCliBot(config)
        
        # Show stats if requested
        if args.stats:
            bot.show_stats()
            sys.exit(0)
        
        # Single question or interactive mode
        if args.question:
            result = bot.ask(args.question)
            print("\n" + "="*50)
            print("📋 RESULT SUMMARY:")
            print(f"Question: {result['question']}")
            print(f"Sources used: {len(result['sources'])}")
            print(f"Answer length: {len(result['answer'])} characters")
        else:
            # Interactive mode
            bot.interactive_mode()
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("\nMake sure you have run:")
        print("  python rag_create_data.py")
        print("\nOr specify correct paths:")
        print(f"  --metadata path/to/metadata.json")
        print(f"  --index path/to/faiss.index")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ============================================================================
# ЗАПУСК
# ============================================================================

if __name__ == "__main__":
    main()
