# =============================================================================
# LLAMMY RAG BUSINESS MODULES - Refactored with Chunking + Document Processing
# llammy_rag_business_modules.py - Drop-in replacement with same API
# =============================================================================

import os
import json
import sqlite3
import hashlib
import requests
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import traceback

# Document processing imports
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("⚠️ Docling not available - install with: pip install docling")

try:
    from llama_index.core import Document as LlamaDocument
    from llama_index.core.node_parser import SentenceSplitter
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    print("⚠️ LlamaIndex not available - install with: pip install llama-index")

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ ChromaDB not available - install with: pip install chromadb")

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("⚠️ PyPDF2 not available - install with: pip install pypdf2")

# =============================================================================
# EMBEDDED MULTI-PURPOSE MODEL (KEPT FROM ORIGINAL)
# =============================================================================

class EmbeddedMultiPurposeModel:
    """Multi-purpose IBM Granite Docling model for validation, context, analysis"""
    
    def __init__(self, model_name="hf.co/danchev/ibm-granite-docling-258M-GGUF:BF16", ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.usage_stats = {
            'validation_calls': 0,
            'context_scoring': 0,
            'code_fixes': 0,
            'rag_queries': 0,
            'vision_assists': 0,
            'api_analysis': 0
        }
        print(f"🔧 Embedded model initialized: {model_name}")
    
    def validate_blender_code(self, code: str, api_version: str = "4.5") -> Dict[str, Any]:
        """Quick Blender code validation with API compatibility"""
        self.usage_stats['validation_calls'] += 1
        
        prompt = f"""Validate Blender {api_version} Python code:
{code[:800]}

Check: syntax errors, deprecated API (object.select, old GPU shaders, seq1/seq2 params), missing imports
Response JSON: {{"valid": bool, "issues": [], "fixes": [], "confidence": float}}"""
        
        return self._call_model_json(prompt, "validation")
    
    def score_context_relevance(self, query: str, content: str, hint: str = "") -> float:
        """Score content relevance for RAG retrieval (0.0-1.0)"""
        self.usage_stats['context_scoring'] += 1
        
        prompt = f"""Score relevance (0.0-1.0):
Query: {query}
Content: {content[:300]}...
Hint: {hint}

Consider: Blender API match, task similarity, code quality
Response: {{"score": float}}"""
        
        result = self._call_model_json(prompt, "scoring")
        return result.get('score', 0.5)
    
    def analyze_api_sophistication(self, content: str) -> Dict[str, Any]:
        """Analyze Blender API sophistication for business value calculation"""
        self.usage_stats['api_analysis'] += 1
        
        prompt = f"""Analyze Blender code sophistication:
{content[:500]}...

Rate: basic ops (1-3), intermediate (4-6), advanced (7-10)
Detect: current API usage, breaking changes, complexity
Response: {{"sophistication_score": int, "api_current": bool, "features": []}}"""
        
        return self._call_model_json(prompt, "api_analysis")
    
    def enhance_rag_query(self, query: str, available_files: List[str]) -> Dict[str, Any]:
        """Enhance RAG query for better retrieval"""
        self.usage_stats['rag_queries'] += 1
        
        prompt = f"""Enhance Blender query:
Query: {query}
Files available: {len(available_files)} total

Extract: intent, keywords, preferred file types, complexity level
Response: {{"intent": str, "keywords": [], "file_types": [], "complexity": str}}"""
        
        return self._call_model_json(prompt, "rag_enhancement")
    
    def _call_model_json(self, prompt: str, task_type: str) -> Dict[str, Any]:
        """Call embedded model expecting JSON response"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_k": 10,
                    "top_p": 0.8,
                    "num_predict": 200
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=8
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '').strip()
                
                try:
                    if '{' in content:
                        json_start = content.find('{')
                        json_end = content.rfind('}') + 1
                        json_str = content[json_start:json_end]
                        return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
                
                return {'raw_response': content, 'task_type': task_type}
            
        except Exception as e:
            print(f"❌ Embedded model error ({task_type}): {e}")
        
        return {'error': 'model_call_failed', 'task_type': task_type}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        total_calls = sum(self.usage_stats.values())
        return {
            'total_calls': total_calls,
            'breakdown': self.usage_stats.copy(),
            'model': self.model_name
        }

# =============================================================================
# INTELLIGENT DOCUMENT CHUNKER
# =============================================================================

class IntelligentDocumentChunker:
    """Handles smart chunking of documents with context preservation"""
    
    def __init__(self, chunk_size=1500, chunk_overlap=200, min_chunk_size=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        if LLAMAINDEX_AVAILABLE:
            self.splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
    
    def chunk_by_semantic_boundaries(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text by semantic boundaries"""
        
        if LLAMAINDEX_AVAILABLE:
            doc = LlamaDocument(text=text, metadata=metadata)
            nodes = self.splitter.get_nodes_from_documents([doc])
            
            chunks = []
            for i, node in enumerate(nodes):
                chunks.append({
                    'text': node.text,
                    'chunk_id': i,
                    'metadata': {**metadata, 'chunk_index': i},
                    'start_char': node.start_char_idx,
                    'end_char': node.end_char_idx
                })
            return chunks
        
        return self._simple_paragraph_chunking(text, metadata)
    
    def _simple_paragraph_chunking(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback paragraph-based chunking"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para.split())
            
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append({
                    'text': '\n\n'.join(current_chunk),
                    'chunk_id': chunk_id,
                    'metadata': {**metadata, 'chunk_index': chunk_id}
                })
                chunk_id += 1
                
                if len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1], para]
                    current_size = len(current_chunk[-1].split()) + para_size
                else:
                    current_chunk = [para]
                    current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        if current_chunk:
            chunks.append({
                'text': '\n\n'.join(current_chunk),
                'chunk_id': chunk_id,
                'metadata': {**metadata, 'chunk_index': chunk_id}
            })
        
        return chunks
    
    def chunk_code_by_function(self, code: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk Python code by function/class definitions"""
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_type = None
        chunk_id = 0
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('def ') or stripped.startswith('class '):
                if current_chunk:
                    chunks.append({
                        'text': '\n'.join(current_chunk),
                        'chunk_id': chunk_id,
                        'metadata': {**metadata, 'chunk_index': chunk_id, 'type': current_type}
                    })
                    chunk_id += 1
                
                current_chunk = [line]
                current_type = 'function' if stripped.startswith('def ') else 'class'
            else:
                current_chunk.append(line)
        
        if current_chunk:
            chunks.append({
                'text': '\n'.join(current_chunk),
                'chunk_id': chunk_id,
                'metadata': {**metadata, 'chunk_index': chunk_id, 'type': current_type or 'code'}
            })
        
        return chunks

# =============================================================================
# =============================================================================
# ENHANCED FILE PROCESSOR WITH DEEP MEDIA EXTRACTION
# =============================================================================

class EnhancedMediaProcessor:
    """Deep content extraction from media files for dataset richness"""
    
    def __init__(self):
        self.pil_available = self._check_pil()
        self.trimesh_available = self._check_trimesh()
        
    def _check_pil(self):
        try:
            from PIL import Image
            return True
        except ImportError:
            return False
    
    def _check_trimesh(self):
        try:
            import trimesh
            return True
        except ImportError:
            return False
    
    def extract_image_data(self, filepath: str, metadata: Dict[str, Any]) -> str:
        """Extract rich data from image files"""
        content_parts = [f"Image File: {metadata['filename']}"]
        
        try:
            if self.pil_available:
                from PIL import Image
                from PIL.ExifTags import TAGS
                
                img = Image.open(filepath)
                
                # Basic info
                content_parts.append(f"Format: {img.format}")
                content_parts.append(f"Dimensions: {img.size[0]}x{img.size[1]} pixels")
                content_parts.append(f"Color Mode: {img.mode}")
                content_parts.append(f"Aspect Ratio: {img.size[0]/img.size[1]:.2f}")
                
                # EXIF data
                exif_data = img._getexif()
                if exif_data:
                    exif_items = []
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag in ['DateTime', 'Make', 'Model', 'Software', 'Copyright']:
                            exif_items.append(f"{tag}: {value}")
                    
                    if exif_items:
                        content_parts.append("EXIF Data:")
                        content_parts.extend(exif_items)
                
                # Color analysis
                if img.mode == 'RGB':
                    img_small = img.resize((50, 50))
                    pixels = list(img_small.getdata())
                    avg_r = sum(p[0] for p in pixels) / len(pixels)
                    avg_g = sum(p[1] for p in pixels) / len(pixels)
                    avg_b = sum(p[2] for p in pixels) / len(pixels)
                    content_parts.append(f"Average Color (RGB): ({avg_r:.0f}, {avg_g:.0f}, {avg_b:.0f})")
                    
                    # Determine dominant color tendency
                    if max(avg_r, avg_g, avg_b) == avg_r:
                        content_parts.append("Color Tendency: Warm/Red tones")
                    elif max(avg_r, avg_g, avg_b) == avg_b:
                        content_parts.append("Color Tendency: Cool/Blue tones")
                    else:
                        content_parts.append("Color Tendency: Green/Natural tones")
            else:
                file_size = os.path.getsize(filepath)
                content_parts.append(f"File Size: {file_size / 1024:.2f} KB")
                
        except Exception as e:
            content_parts.append(f"Extraction Error: {str(e)}")
        
        return "\n".join(content_parts)
    
    def extract_3d_model_data(self, filepath: str, metadata: Dict[str, Any]) -> str:
        """Extract rich data from 3D model files"""
        content_parts = [f"3D Model: {metadata['filename']}"]
        
        try:
            ext = metadata['extension'].lower()
            
            if self.trimesh_available and ext in ['.obj', '.stl', '.gltf', '.glb']:
                import trimesh
                
                mesh = trimesh.load(filepath)
                
                if isinstance(mesh, trimesh.Scene):
                    # Multiple meshes
                    content_parts.append(f"Type: Scene with {len(mesh.geometry)} objects")
                    total_vertices = sum(geom.vertices.shape[0] for geom in mesh.geometry.values())
                    total_faces = sum(geom.faces.shape[0] for geom in mesh.geometry.values())
                else:
                    # Single mesh
                    content_parts.append("Type: Single mesh")
                    total_vertices = mesh.vertices.shape[0]
                    total_faces = mesh.faces.shape[0]
                
                content_parts.append(f"Vertices: {total_vertices:,}")
                content_parts.append(f"Faces: {total_faces:,}")
                
                # Complexity assessment
                if total_vertices < 1000:
                    content_parts.append("Complexity: Low-poly")
                elif total_vertices < 10000:
                    content_parts.append("Complexity: Medium-poly")
                elif total_vertices < 100000:
                    content_parts.append("Complexity: High-poly")
                else:
                    content_parts.append("Complexity: Very high-poly")
                
                # Bounds
                if not isinstance(mesh, trimesh.Scene):
                    bounds = mesh.bounds
                    dimensions = bounds[1] - bounds[0]
                    content_parts.append(f"Dimensions: {dimensions[0]:.2f} x {dimensions[1]:.2f} x {dimensions[2]:.2f} units")
                    content_parts.append(f"Volume: {mesh.volume:.2f} cubic units")
                    content_parts.append(f"Surface Area: {mesh.area:.2f} square units")
                    
                    # Check for watertightness
                    if mesh.is_watertight:
                        content_parts.append("Mesh Status: Watertight (3D printable)")
                    else:
                        content_parts.append("Mesh Status: Non-watertight (may have holes)")
            
            elif ext == '.fbx':
                # FBX requires specialized parsing - basic metadata only
                file_size = os.path.getsize(filepath)
                content_parts.append(f"Format: Autodesk FBX")
                content_parts.append(f"File Size: {file_size / (1024*1024):.2f} MB")
                content_parts.append("Note: FBX files may contain animations, materials, and rigging")
            
            else:
                file_size = os.path.getsize(filepath)
                content_parts.append(f"File Size: {file_size / 1024:.2f} KB")
                
        except Exception as e:
            content_parts.append(f"Extraction Error: {str(e)}")
            file_size = os.path.getsize(filepath)
            content_parts.append(f"File Size: {file_size / 1024:.2f} KB")
        
        return "\n".join(content_parts)
    
    def extract_audio_data(self, filepath: str, metadata: Dict[str, Any]) -> str:
        """Extract rich data from audio files"""
        content_parts = [f"Audio File: {metadata['filename']}"]
        
        try:
            import wave
            
            ext = metadata['extension'].lower()
            
            if ext == '.wav':
                with wave.open(filepath, 'rb') as wav_file:
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    framerate = wav_file.getframerate()
                    frames = wav_file.getnframes()
                    
                    duration = frames / float(framerate)
                    
                    content_parts.append(f"Format: WAV (Uncompressed)")
                    content_parts.append(f"Duration: {duration:.2f} seconds")
                    content_parts.append(f"Sample Rate: {framerate} Hz")
                    content_parts.append(f"Channels: {channels} ({'Stereo' if channels == 2 else 'Mono'})")
                    content_parts.append(f"Bit Depth: {sample_width * 8} bit")
                    
                    # Quality assessment
                    if framerate >= 44100:
                        content_parts.append("Quality: CD Quality or better")
                    elif framerate >= 22050:
                        content_parts.append("Quality: Standard")
                    else:
                        content_parts.append("Quality: Low")
            
            elif ext == '.mp3':
                file_size = os.path.getsize(filepath)
                content_parts.append(f"Format: MP3 (Compressed)")
                content_parts.append(f"File Size: {file_size / (1024*1024):.2f} MB")
                
                # Estimate duration based on typical bitrates
                estimated_duration = (file_size * 8) / (128 * 1024)  # Assuming 128kbps
                content_parts.append(f"Estimated Duration: ~{estimated_duration:.0f} seconds")
            
        except Exception as e:
            content_parts.append(f"Extraction Error: {str(e)}")
            file_size = os.path.getsize(filepath)
            content_parts.append(f"File Size: {file_size / (1024*1024):.2f} MB")
        
        return "\n".join(content_parts)
    
    def extract_video_data(self, filepath: str, metadata: Dict[str, Any]) -> str:
        """Extract rich data from video files"""
        content_parts = [f"Video File: {metadata['filename']}"]
        
        try:
            file_size = os.path.getsize(filepath)
            content_parts.append(f"Format: MP4")
            content_parts.append(f"File Size: {file_size / (1024*1024):.2f} MB")
            
            # For more detailed extraction, would need opencv or ffmpeg
            # For now, provide basic metadata
            
            # Estimate based on file size and typical bitrates
            estimated_duration = (file_size * 8) / (5 * 1024 * 1024)  # Assuming 5Mbps
            content_parts.append(f"Estimated Duration: ~{estimated_duration:.0f} seconds")
            content_parts.append("Note: Contains video and possibly audio streams")
            
        except Exception as e:
            content_parts.append(f"Extraction Error: {str(e)}")
        
        return "\n".join(content_parts)

# =============================================================================
# UPDATE ChunkedFileProcessor
# =============================================================================

class ChunkedFileProcessor:
    """Processes various file types with intelligent chunking"""
    
    def __init__(self, chunker: IntelligentDocumentChunker):
        self.chunker = chunker
        self.docling_converter = DocumentConverter() if DOCLING_AVAILABLE else None
        self.media_processor = EnhancedMediaProcessor()
        
        self.supported_formats = {
            '.py': self._process_python,
            '.txt': self._process_text,
            '.md': self._process_markdown,
            '.pdf': self._process_pdf,
            '.blend': self._process_blend,
            '.json': self._process_json,
            '.csv': self._process_csv,
            
            # Media files with deep extraction
            '.jpg': self._process_image,
            '.jpeg': self._process_image,
            '.png': self._process_image,
            '.mp3': self._process_audio,
            '.wav': self._process_audio,
            '.mp4': self._process_video,
            
            # 3D models with deep extraction
            '.obj': self._process_3d_model,
            '.fbx': self._process_3d_model,
            '.gltf': self._process_3d_model,
            '.glb': self._process_3d_model,
            '.stl': self._process_3d_model,
        }
    
    # ... existing methods stay the same ...
    
    def _process_csv(self, filepath: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process CSV files with structure analysis"""
        try:
            import csv
            
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if not rows:
                return {'success': False, 'error': 'Empty CSV file'}
            
            # Build descriptive text
            text_parts = [f"CSV File: {metadata['filename']}"]
            text_parts.append(f"Rows: {len(rows)}")
            text_parts.append(f"Columns: {len(rows[0]) if rows else 0}")
            
            # Header row
            if rows:
                text_parts.append(f"Headers: {', '.join(rows[0])}")
            
            # Sample data (first 5 rows)
            text_parts.append("\nSample Data:")
            for i, row in enumerate(rows[:6], 1):  # Header + 5 data rows
                text_parts.append(f"Row {i}: {', '.join(str(cell) for cell in row)}")
            
            text = "\n".join(text_parts)
            
            chunks = self.chunker.chunk_by_semantic_boundaries(text, metadata)
            
            return {'success': True, 'total_chunks': len(chunks), 'chunks': chunks, 'metadata': metadata}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _process_image(self, filepath: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process image with deep extraction"""
        try:
            text = self.media_processor.extract_image_data(filepath, metadata)
            
            chunks = [{
                'text

# =============================================================================
# CHUNKED VECTOR STORE
# =============================================================================

class ChunkedVectorStore:
    """Store and retrieve chunked documents"""
    
    def __init__(self, db_path: str, collection_name: str = "llammy_chunks"):
        self.db_path = db_path
        self.collection_name = collection_name
        
        if CHROMADB_AVAILABLE:
            self.client = chromadb.PersistentClient(path=db_path)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.use_chromadb = True
        else:
            self.use_chromadb = False
            self._init_sqlite_fallback()
    
    def _init_sqlite_fallback(self):
        """Initialize SQLite as fallback"""
        os.makedirs(self.db_path, exist_ok=True)
        conn = sqlite3.connect(os.path.join(self.db_path, 'chunks.db'))
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                filepath TEXT,
                chunk_index INTEGER,
                text TEXT,
                metadata TEXT,
                content_hash TEXT,
                timestamp TEXT
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_filepath ON chunks(filepath)')
        conn.commit()
        conn.close()
    
    def store_chunks(self, filepath: str, chunks: List[Dict[str, Any]]) -> bool:
        """Store document chunks"""
        try:
            if self.use_chromadb:
                ids = []
                texts = []
                metadatas = []
                
                for chunk in chunks:
                    chunk_id = f"{filepath}::{chunk['chunk_id']}"
                    ids.append(chunk_id)
                    texts.append(chunk['text'])
                    metadatas.append(chunk.get('metadata', {}))
                
                self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
            else:
                conn = sqlite3.connect(os.path.join(self.db_path, 'chunks.db'))
                for chunk in chunks:
                    chunk_id = f"{filepath}::{chunk['chunk_id']}"
                    content_hash = hashlib.sha256(chunk['text'].encode()).hexdigest()
                    
                    conn.execute('''
                        INSERT OR REPLACE INTO chunks 
                        (id, filepath, chunk_index, text, metadata, content_hash, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        chunk_id, filepath, chunk['chunk_id'], chunk['text'],
                        json.dumps(chunk.get('metadata', {})), content_hash,
                        datetime.now().isoformat()
                    ))
                conn.commit()
                conn.close()
            
            return True
        except Exception as e:
            print(f"Error storing chunks: {e}")
            return False
    
    def search_chunks(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        try:
            if self.use_chromadb:
                results = self.collection.query(query_texts=[query], n_results=top_k)
                
                chunks = []
                for i, doc in enumerate(results['documents'][0]):
                    chunks.append({
                        'text': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else 0
                    })
                return chunks
            else:
                conn = sqlite3.connect(os.path.join(self.db_path, 'chunks.db'))
                cursor = conn.execute('''
                    SELECT text, metadata FROM chunks
                    WHERE text LIKE ?
                    LIMIT ?
                ''', (f'%{query}%', top_k))
                
                chunks = []
                for row in cursor.fetchall():
                    chunks.append({'text': row[0], 'metadata': json.loads(row[1])})
                conn.close()
                return chunks
        except Exception as e:
            print(f"Error searching chunks: {e}")
            return []

# =============================================================================
# BLENDER VERSION ADVANTAGE (KEPT FROM ORIGINAL)
# =============================================================================

class BlenderVersionAdvantage:
    """Competitive analysis for Blender API versions"""
    
    def __init__(self):
        self.version_multipliers = {
            '4.5': 50.0,
            '4.4': 15.0,
            '4.3': 8.0,
            '4.2': 4.0,
            '4.1': 2.0,
            'legacy': 1.0
        }
    
    def calculate_competitive_value(self, content: str, embedded_model: EmbeddedMultiPurposeModel = None) -> Dict[str, Any]:
        """Calculate competitive business value"""
        base_value = 1.0
        multiplier_factors = []
        
        if '.blend' in content:
            base_value *= 15.0
            multiplier_factors.append("Native .blend format: 15x")
        elif 'import bpy' in content:
            base_value *= 10.0
            multiplier_factors.append("Blender Python script: 10x")
        
        detected_version = self.detect_version(content)
        version_multiplier = self.version_multipliers.get(detected_version, 1.0)
        base_value *= version_multiplier
        multiplier_factors.append(f"API version {detected_version}: {version_multiplier}x")
        
        if embedded_model:
            api_analysis = embedded_model.analyze_api_sophistication(content)
            sophistication = api_analysis.get('sophistication_score', 5)
            if sophistication >= 7:
                base_value *= 2.0
                multiplier_factors.append("Advanced API usage: 2x")
        
        return {
            'business_value': base_value,
            'competitive_advantage': base_value > 10.0,
            'multiplier_factors': multiplier_factors,
            'api_version': detected_version
        }
    
    def detect_version(self, content: str) -> str:
        """Detect Blender API version"""
        if any(p in content for p in ['input1=', 'input2=', 'GeometrySet', 'gpu.shader.create_from_info']):
            return '4.5'
        if any(p in content for p in ['select_set(True)', 'geometry_nodes']):
            return '4.4'
        if any(p in content for p in ['.select = True', '.select = False']):
            return 'legacy'
        return '4.3'

# =============================================================================
# RAG BUSINESS INTEGRATION (REFACTORED WITH CHUNKING)
# =============================================================================

class RAGBusinessIntegration:
    """Main RAG interface with chunking - SAME API as original"""
    
    def __init__(self, embedded_model_name: str = "hf.co/danchev/ibm-granite-docling-258M-GGUF:BF16"):
        self.embedded_model = EmbeddedMultiPurposeModel(embedded_model_name)
        self.business_analyzer = BlenderVersionAdvantage()
        
        # Initialize chunking components
        storage_path = os.path.expanduser("~/.llammy/rag_storage")
        self.chunker = IntelligentDocumentChunker(chunk_size=1500, chunk_overlap=200)
        self.file_processor = ChunkedFileProcessor(self.chunker)
        self.vector_store = ChunkedVectorStore(storage_path)
        
        self.stats = {
            'files_processed': 0,
            'total_chunks': 0,
            'queries_handled': 0
        }
        
        print(f"🚀 RAG Business Integration initialized with {embedded_model_name}")
        print(f"   Chunking: Enabled")
        print(f"   Vector Store: {'ChromaDB' if self.vector_store.use_chromadb else 'SQLite'}")
    
    def ingest_file(self, filepath: str) -> Dict[str, Any]:
        """Ingest a file with chunking"""
        result = self.file_processor.process_file(filepath)
        
        if not result.get('success'):
            return result
        
        chunks = result.get('chunks', [])
        stored = self.vector_store.store_chunks(filepath, chunks)
        
        if stored:
            self.stats['files_processed'] += 1
            self.stats['total_chunks'] += len(chunks)
        
        return result
    
    def ingest_directory(self, directory: str, recursive: bool = True) -> Dict[str, Any]:
        """Ingest all supported files in a directory"""
        dir_path = Path(directory)
        if not dir_path.exists():
            return {'success': False, 'error': 'Directory not found'}
        
        pattern = '**/*' if recursive else '*'
        files = []
        results = []
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix in self.file_processor.supported_formats:
                files.append(str(file_path))
        
        for filepath in files:
            result = self.ingest_file(filepath)
            results.append(result)
        
        successful = sum(1 for r in results if r.get('success'))
        
        return {
            'success': True,
            'total_files': len(files),
            'successful': successful,
            'failed': len(files) - successful
        }
    
    def get_context_for_request(self, user_request: str, enhanced_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main RAG context retrieval - SAME API as original"""
        self.stats['queries_handled'] += 1
        
        # Search for relevant chunks
        chunks = self.vector_store.search_chunks(user_request, top_k=5)
        
        if not chunks:
            return {
                'context': '',
                'business_value': 0,
                'competitive_files': 0,
                'embedded_model_stats': self.embedded_model.get_stats()
            }
        
        # Build context with relevance scoring
        context_parts = []
        total_business_value = 0
        
        for chunk in chunks:
            relevance = self.embedded_model.score_context_relevance(
                user_request, chunk['text'], ""
            )
            
            if relevance > 0.3:
                metadata = chunk.get('metadata', {})
                filepath = metadata.get('filepath', 'unknown')
                context_parts.append(f"[{Path(filepath).name}]\n{chunk['text']}")
                
                # Calculate business value
                value_analysis = self.business_analyzer.calculate_competitive_value(
                    chunk['text'], self.embedded_model
                )
                total_business_value += value_analysis['business_value']
        
        context = "\n---\n".join(context_parts)
        
        return {
            'context': context,
            'business_value': total_business_value,
            'competitive_files': len(chunks),
            'embedded_model_stats': self.embedded_model.get_stats()
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive RAG system status"""
        return {
            'embedded_model': self.embedded_model.get_stats(),
            'chunking': 'enabled',
            'vector_store': 'ChromaDB' if self.vector_store.use_chromadb else 'SQLite',
            'files_processed': self.stats['files_processed'],
            'total_chunks': self.stats['total_chunks'],
            'queries_handled': self.stats['queries_handled']
        }

# =============================================================================
# GLOBAL INSTANCE (KEPT FROM ORIGINAL)
# =============================================================================

rag_business_system = None

def get_rag_business_system(embedded_model: str = "hf.co/danchev/ibm-granite-docling-258M-GGUF:BF16") -> RAGBusinessIntegration:
    """Get global RAG business system"""
    global rag_business_system
    if rag_business_system is None:
        rag_business_system = RAGBusinessIntegration(embedded_model)
    return rag_business_system

if __name__ == "__main__":
    print("RAG BUSINESS MODULES with Chunking + Document Processing LOADED!")
    print("Features:")
    print("  - IBM Granite Docling 258M integration")
    print("  - Intelligent document chunking")
    print("  - Multi-format support: .py, .pdf, .md, .txt, .json, .blend")
    print("  - Vector storage (ChromaDB/SQLite)")
    print("  - Business value tracking")
  
