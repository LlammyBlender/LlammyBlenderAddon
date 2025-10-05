# =============================================================================
# LLAMMY BLENDER ADDON - Complete Integrated __init__.py v6.0
# Enhanced modular dual AI with Spatial Intelligence and MCP architecture
# =============================================================================
# LLAMMY BLENDER ADDON - Re-engineered __init__.py for MCP Architecture
# =============================================================================

import bpy
import os
import time
import sys
import json
import asyncio
import websockets
import threading
import queue
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import StringProperty, BoolProperty, IntProperty


# =============================================================================
# ADDON INFORMATION
# =============================================================================
# This bl_info has been updated to reflect the new MCP architecture
bl_info = {
    "name": "Llammy MCP AI",
    "author": "Llammy Development Team",
    "version": (5, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Llammy",
    "description": "Modular dual AI for end-to-end animation with self-debugging",
    "category": "Animation",
}

# =============================================================================
# IMPORT CORE MODULES
# =============================================================================

try:
    # Import core systems
    from . import llammy_core
    from . import llammy_ai
    from . import llammy_rag_business_modules
    from . import llammy_spatial_intelligence
    
    # Import successful - all modules available
    MODULES_LOADED = True
    MISSING_MODULES = []
    
except ImportError as e:
    MODULES_LOADED = False
    MISSING_MODULES = [str(e)]
    print(f"Module import failed: {e}")

# =============================================================================
# DEBUG AND LOGGING SYSTEM
# =============================================================================
class DebugManager:
    """Centralized debug and logging system"""
    
    def __init__(self):
        self.logs = []
        self.max_logs = 100
        
    def log(self, level: str, message: str):
        """Log a message with a timestamp and level"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        log_entry = f"[{timestamp}] [{level.upper()}] {message}"
        self.logs.append(log_entry)
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

# Instantiate the debug manager
debug = DebugManager()


# =============================================================================
# MCP BLENDER BRIDGE - WEBSOCKET CLIENT
# This is the new core communication module
# =============================================================================

class MCPBlenderBridge:
    """
    A thread-safe WebSocket bridge to the MCP core server.
    All communication between Blender and the core happens here.
    """
    
    def __init__(self, host="localhost", port=8765):
        self.uri = f"ws://{host}:{port}"
        self.connected = False
        self.ws = None
        self.thread = None
        self.task_queue = queue.Queue()
        self.last_heartbeat = None
        self.last_sync_time = None
        self.scene_state = None
        self.connection_lock = threading.Lock()
        
    async def connect(self):
        """Establish a WebSocket connection to the server"""
        if self.connected:
            debug.log("INFO", "Already connected to the MCP server.")
            return

        debug.log("INFO", f"Attempting to connect to MCP server at {self.uri}")
        try:
            self.ws = await websockets.connect(self.uri, open_timeout=5)
            self.connected = True
            debug.log("INFO", "✅ Connected to MCP server.")
            self.thread = threading.Thread(target=asyncio.run, args=(self.listen(),))
            self.thread.daemon = True
            self.thread.start()
            
            # Update UI properties
            bpy.context.scene.llammy_mcp_props.is_connected = True
            bpy.context.scene.llammy_mcp_props.status_message = "Connected to MCP Server"
            bpy.context.scene.llammy_mcp_props.last_heartbeat = time.strftime('%Y-%m-%d %H:%M:%S')

            await self.send_command('ping')
            
        except (websockets.exceptions.ConnectionRefusedError, asyncio.TimeoutError) as e:
            debug.log("ERROR", f"Connection failed: {e}")
            self.connected = False
            bpy.context.scene.llammy_mcp_props.is_connected = False
            bpy.context.scene.llammy_mcp_props.status_message = f"Connection failed: {e}"

    async def disconnect(self):
        """Close the WebSocket connection"""
        if not self.connected:
            debug.log("INFO", "Already disconnected.")
            return

        debug.log("INFO", "Disconnecting from MCP server...")
        with self.connection_lock:
            self.connected = False
            if self.ws:
                await self.ws.close()
            
            # Update UI properties
            bpy.context.scene.llammy_mcp_props.is_connected = False
            bpy.context.scene.llammy_mcp_props.status_message = "Disconnected"
            bpy.context.scene.llammy_mcp_props.last_heartbeat = "N/A"

    async def listen(self):
        """Listen for messages from the server"""
        try:
            while self.connected:
                message = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                self.handle_message(json.loads(message))
        except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
            debug.log("WARNING", "Connection to server lost. Attempting to reconnect...")
            await self.disconnect()
            await self.connect()
        except Exception as e:
            debug.log("ERROR", f"Error in listener thread: {e}")
            await self.disconnect()

    def handle_message(self, message: Dict[str, Any]):
        """Handle incoming messages and add tasks to the queue"""
        msg_type = message.get("type")
        
        if msg_type == "heartbeat":
            self.last_heartbeat = time.time()
            bpy.context.scene.llammy_mcp_props.last_heartbeat = time.strftime('%Y-%m-%d %H:%M:%S')
            debug.log("INFO", "Received heartbeat.")
        elif msg_type == "execute":
            # Add command to thread-safe queue for Blender's main thread to process
            self.task_queue.put(message.get("data"))
        elif msg_type == "state_change":
            self.scene_state = message.get("data")
            self.last_sync_time = time.time()
            debug.log("INFO", "Received state change update.")
        else:
            debug.log("WARNING", f"Unknown message type: {msg_type}")

    async def send_command(self, command: str, data: Optional[Dict[str, Any]] = None):
        """Send a command to the server"""
        if self.connected and self.ws:
            try:
                message = {
                    "type": "command",
                    "command": command,
                    "data": data if data else {}
                }
                await self.ws.send(json.dumps(message))
                debug.log("INFO", f"Sent command: {command}")
            except websockets.exceptions.ConnectionClosed:
                debug.log("ERROR", "Failed to send command: Connection closed.")
                await self.disconnect()
            except Exception as e:
                debug.log("ERROR", f"Failed to send command: {e}")
    
    def process_queue(self):
        """Process the thread-safe queue from Blender's main thread"""
        while not self.task_queue.empty():
            task = self.task_queue.get()
            debug.log("INFO", f"Executing task from queue: {task}")
            # This is where your code from llammy_ai.py and others would execute
            # You would need to add a system to map `task` to a function call.
            self.task_queue.task_done()

# Instantiate the MCP bridge
mcp_bridge = MCPBlenderBridge()

# =============================================================================
# BLENDER UI PROPERTIES
# This adds a dedicated property group to store and manage UI metrics
# =============================================================================

class LlammyMCPProperties(bpy.types.PropertyGroup):
    """Properties to store and display MCP status in the UI"""
    is_connected: BoolProperty(name="Connected", default=False)
    status_message: StringProperty(name="Status", default="Disconnected")
    last_heartbeat: StringProperty(name="Last Heartbeat", default="N/A")

# =============================================================================
# INTEGRATED MODEL MANAGEMENT
# =============================================================================

class IntegratedModelManager:
    """Integrated model management with spatial intelligence support"""
    
    def __init__(self):
        self.cached_models = []
        self.cache_timestamp = 0
        self.cache_duration = 300  # 5 minutes
        self.ollama_url = "http://localhost:11434"
        self.connection_status = "unknown"
    
    def get_available_models(self) -> List[Tuple[str, str, str]]:
        """Get available models with caching"""
        current_time = time.time()
        
        # Use cache if recent
        if (current_time - self.cache_timestamp) < self.cache_duration and self.cached_models:
            return self.cached_models
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                self.connection_status = "connected"
                
                if models:
                    items = []
                    
                    # Prioritize specialized models
                    triangle_models = [m for m in models if 'triangle104' in m['name'].lower()]
                    llammy_models = [m for m in models if any(term in m['name'].lower() for term in ['llammy', 'sentient', 'nemotron'])]
                    standard_models = [m for m in models if m not in triangle_models + llammy_models]
                    
                    for model in triangle_models + llammy_models + standard_models:
                        name = model['name']
                        size_info = model.get('size', 0)
                        size_gb = size_info / (1024**3) if size_info else 0
                        
                        if 'triangle104' in name.lower():
                            display = f"🔥 {name} ({size_gb:.1f}GB)"
                            desc = f"Triangle104 Premium: {name}"
                        elif any(term in name.lower() for term in ['llammy', 'sentient', 'nemotron']):
                            display = f"⚡ {name} ({size_gb:.1f}GB)"
                            desc = f"Specialized Blender: {name}"
                        elif any(term in name.lower() for term in ['llama', 'qwen', 'gemma']):
                            display = f"🤖 {name} ({size_gb:.1f}GB)"
                            desc = f"Standard Model: {name}"
                        else:
                            display = f"📦 {name} ({size_gb:.1f}GB)"
                            desc = f"Other Model: {name}"
                        
                        items.append((name, display, desc))
                    
                    self.cached_models = items
                    self.cache_timestamp = current_time
                    return items
        
        except requests.exceptions.ConnectionError:
            self.connection_status = "disconnected"
            debug.log("WARNING", "Ollama connection failed")
        except requests.exceptions.RequestException as e:
            self.connection_status = f"error: {str(e)[:20]}"
            debug.log("ERROR", f"Model enumeration error: {e}")
        except Exception as e:
            self.connection_status = "unknown error"
            debug.log("ERROR", f"Unexpected model enumeration error: {e}")
        
        # Fallback models
        fallback_models = [
            ("llama3.2:3b", "🤖 llama3.2:3b (Offline)", "Fallback - Ollama not connected"),
            ("qwen2.5:3b", "🤖 qwen2.5:3b (Offline)", "Fallback - Ollama not connected"),
            ("no_models", "⚠ No Models Available", "Check Ollama connection"),
        ]
        
        if not self.cached_models:
            self.cached_models = fallback_models
            self.cache_timestamp = current_time
        
        return self.cached_models
    
    def refresh_models(self):
        """Force refresh of model cache"""
        self.cache_timestamp = 0
        self.cached_models = []
        return self.get_available_models()
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status"""
        return {
            'status': self.connection_status,
            'cached_models': len(self.cached_models),
            'cache_age': time.time() - self.cache_timestamp,
            'ollama_url': self.ollama_url
        }

# Global model manager
model_manager = IntegratedModelManager()

def get_models_for_properties(scene, context):
    """Get models for Blender properties"""
    return model_manager.get_available_models()

# =============================================================================
# ENHANCED FILE MANAGER
# =============================================================================

class EnhancedFileManager:
    """Enhanced file manager with folder drop support"""
    
    def __init__(self):
        self.selected_files = []
        self.supported_extensions = {
            '.py': 'Python',
            '.blend': 'Blender',
            '.json': 'JSON',
            '.txt': 'Text',
            '.csv': 'CSV',
            '.md': 'Markdown',
            '.xml': 'XML',
            '.yaml': 'YAML',
            '.yml': 'YAML'
        }
        self.stats = {
            'total_files': 0,
            'total_size_mb': 0.0,
            'file_types': {},
            'last_update': time.time(),
            'largest_file': '',
            'newest_file': ''
        }
    
    def add_files(self, file_paths: List[str]):
        """Add files with validation and stats update"""
        added_count = 0
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                if os.path.isdir(file_path):
                    added_count += self._add_directory(file_path)
                else:
                    if self._is_supported_file(file_path) and file_path not in self.selected_files:
                        self.selected_files.append(file_path)
                        added_count += 1
        
        if added_count > 0:
            self._update_stats()
            debug.log("INFO", f"Added {added_count} files/folders")
    
    def _add_directory(self, dir_path: str) -> int:
        """Recursively add files from directory"""
        added_count = 0
        try:
            for root, dirs, files in os.walk(dir_path):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'node_modules'}]
                
                for file in files:
                    if file.startswith('.'):
                        continue
                    
                    file_path = os.path.join(root, file)
                    if self._is_supported_file(file_path) and file_path not in self.selected_files:
                        self.selected_files.append(file_path)
                        added_count += 1
                        
                        if added_count >= 500:
                            debug.log("WARNING", f"File limit reached: {added_count} files added")
                            break
                
                if added_count >= 500:
                    break
                    
        except Exception as e:
            debug.log("ERROR", f"Error processing directory {dir_path}: {e}")
        
        return added_count
    
    def _is_supported_file(self, file_path: str) -> bool:
        """Check if file type is supported"""
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self.supported_extensions
    
    def _update_stats(self):
        """Update comprehensive file statistics"""
        self.stats = {
            'total_files': len(self.selected_files),
            'total_size_mb': 0.0,
            'file_types': {},
            'last_update': time.time(),
            'largest_file': '',
            'largest_size': 0,
            'newest_file': '',
            'newest_time': 0
        }
        
        for file_path in self.selected_files:
            try:
                stat = os.stat(file_path)
                size = stat.st_size
                mtime = stat.st_mtime
                
                self.stats['total_size_mb'] += size / (1024 * 1024)
                
                if size > self.stats['largest_size']:
                    self.stats['largest_size'] = size
                    self.stats['largest_file'] = os.path.basename(file_path)
                
                if mtime > self.stats['newest_time']:
                    self.stats['newest_time'] = mtime
                    self.stats['newest_file'] = os.path.basename(file_path)
                
                ext = os.path.splitext(file_path)[1].lower()
                type_name = self.supported_extensions.get(ext, ext)
                self.stats['file_types'][type_name] = self.stats['file_types'].get(type_name, 0) + 1
                
            except Exception:
                pass
    
    def clear_files(self):
        """Clear all files and reset stats"""
        count = len(self.selected_files)
        self.selected_files.clear()
        self._update_stats()
        debug.log("INFO", f"Cleared {count} files")
    
    def get_files_for_context(self) -> List[str]:
        return self.selected_files.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        return self.stats.copy()

# Global file manager
file_manager = EnhancedFileManager()

# =============================================================================
# INTEGRATED MODULE ORCHESTRATOR
# =============================================================================

class IntegratedModuleOrchestrator:
    """Integrated module orchestrator with all systems"""
    
    def __init__(self):
        self.initialization_status = {
            'core_initialized': False,
            'spatial_intelligence_active': False,
            'rag_business_active': False,
            'dual_ai_active': False,
            'dependencies_ok': MODULES_LOADED
        }
        
        # Core systems
        self.llammy_core = None
        self.spatial_intelligence = None
        self.rag_system = None
        self.dual_ai_engine = None
        
        # Statistics
        self.ai_stats = {
            'requests_processed': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'total_tokens_processed': 0,
            'spatial_enhancements': 0
        }
        
        self.system_stats = {
            'uptime': time.time(),
            'memory_usage': 0,
            'active_processes': 0,
            'error_count': 0
        }
    
    def initialize(self):
        """Initialize all integrated systems"""
        try:
            if not MODULES_LOADED:
                debug.log("ERROR", f"Cannot initialize - missing modules: {MISSING_MODULES}")
                return False
            
            debug.log("INFO", "Initializing integrated Llammy system...")
            
            # Initialize core system
            try:
                init_result = llammy_core.initialize_llammy_system()
                if init_result.get('success'):
                    self.llammy_core = llammy_core.get_llammy_core()
                    self.initialization_status['core_initialized'] = True
                    debug.log("INFO", "Core system initialized")
                else:
                    debug.log("WARNING", f"Core initialization warning: {init_result.get('error', 'Unknown')}")
            except Exception as e:
                debug.log("WARNING", f"Core initialization failed: {e}")
            
            # Initialize Spatial Intelligence
            try:
                spatial_result = llammy_spatial_intelligence.initialize_spatial_intelligence(
                    self.llammy_core, file_manager, model_manager
                )
                if spatial_result.get('success'):
                    self.spatial_intelligence = llammy_spatial_intelligence.get_spatial_intelligence_module()
                    self.initialization_status['spatial_intelligence_active'] = True
                    debug.log("INFO", f"Spatial Intelligence initialized: {spatial_result.get('scene_objects', 0)} objects analyzed")
                else:
                    debug.log("WARNING", f"Spatial Intelligence warning: {spatial_result.get('error', 'Unknown')}")
            except Exception as e:
                debug.log("WARNING", f"Spatial Intelligence initialization failed: {e}")
            
            # Initialize RAG Business System
            try:
                self.rag_system = llammy_rag_business_modules.get_rag_business_system()
                if self.rag_system:
                    self.initialization_status['rag_business_active'] = True
                    debug.log("INFO", "RAG Business system initialized")
            except Exception as e:
                debug.log("WARNING", f"RAG system initialization failed: {e}")
            
            # Initialize Dual AI Engine
            try:
                self.dual_ai_engine = llammy_ai.get_ai_engine()
                if self.dual_ai_engine:
                    # Connect services
                    self.dual_ai_engine.set_services(
                        self.llammy_core.ollama_service if self.llammy_core else None,
                        self.llammy_core.vision_service if self.llammy_core else None
                    )
                    self.initialization_status['dual_ai_active'] = True
                    debug.log("INFO", "Dual AI engine initialized")
            except Exception as e:
                debug.log("WARNING", f"Dual AI initialization failed: {e}")
            
            # Update system stats
            self.system_stats['active_processes'] = sum(1 for status in self.initialization_status.values() if status)
            
            debug.log("INFO", "Integrated system initialization complete")
            return True
            
        except Exception as e:
            self.system_stats['error_count'] += 1
            debug.log("ERROR", f"System initialization failed: {e}")
            return False
    
    def execute_integrated_request(self, user_request: str, creative_model: str = None,
                                  technical_model: str = None, context_files: List[str] = None) -> Dict[str, Any]:
        """Execute request through integrated pipeline"""
        
        start_time = time.time()
        self.ai_stats['requests_processed'] += 1
        
        try:
            # Step 1: Get enhanced context from RAG system
            enhanced_context = ""
            if self.rag_system and context_files:
                try:
                    rag_context = self.rag_system.get_context_for_request(user_request)
                    enhanced_context = rag_context.get('context', '')
                    debug.log("INFO", f"RAG context: {len(enhanced_context)} characters")
                except Exception as e:
                    debug.log("WARNING", f"RAG context failed: {e}")
            
            # Step 2: Get spatial intelligence enhancement
            spatial_enhancement = {}
            if self.spatial_intelligence:
                try:
                    spatial_result = llammy_spatial_intelligence.enhance_ai_request_with_spatial_context(
                        user_request, {}
                    )
                    if spatial_result.get('success'):
                        spatial_enhancement = spatial_result
                        self.ai_stats['spatial_enhancements'] += 1
                        debug.log("INFO", "Spatial intelligence applied")
                except Exception as e:
                    debug.log("WARNING", f"Spatial enhancement failed: {e}")
            
            # Step 3: Execute through dual AI with enhancements
            if self.dual_ai_engine:
                try:
                    ai_result = self.dual_ai_engine.execute_request(
                        user_request,
                        enhanced_context,
                        creative_model,
                        technical_model,
                        spatial_enhancement
                    )
                    
                    processing_time = time.time() - start_time
                    
                    if ai_result.get('success'):
                        self.ai_stats['successful_requests'] += 1
                        
                        # Update average response time
                        self._update_avg_response_time(processing_time)
                        
                        return {
                            'success': True,
                            'generated_code': ai_result.get('generated_code', ''),
                            'processing_time': processing_time,
                            'method': 'integrated_pipeline',
                            'enhancements_used': {
                                'rag_context': bool(enhanced_context),
                                'spatial_intelligence': bool(spatial_enhancement.get('spatial_intelligence_applied')),
                                'dual_ai': True
                            },
                            'models_used': ai_result.get('models_used', {}),
                            'creative_analysis': ai_result.get('creative_analysis', ''),
                            'spatial_context': spatial_enhancement.get('semantic_context', {})
                        }
                    else:
                        raise Exception(ai_result.get('error', 'AI processing failed'))
                        
                except Exception as e:
                    debug.log("ERROR", f"AI execution failed: {e}")
                    raise e
            else:
                raise Exception("Dual AI engine not available")
        
        except Exception as e:
            processing_time = time.time() - start_time
            self.ai_stats['failed_requests'] += 1
            self._update_avg_response_time(processing_time)
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time,
                'method': 'integrated_pipeline_failed',
                'generated_code': f'# Error: {e}'
            }
    
    def _update_avg_response_time(self, processing_time: float):
        """Update average response time"""
        total_requests = self.ai_stats['requests_processed']
        current_avg = self.ai_stats['avg_response_time']
        
        if total_requests > 0:
            new_avg = ((current_avg * (total_requests - 1)) + processing_time) / total_requests
            self.ai_stats['avg_response_time'] = new_avg
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        current_time = time.time()
        uptime_seconds = current_time - self.system_stats['uptime']
        
        return {
            'initialization_status': self.initialization_status,
            'modules_loaded': MODULES_LOADED,
            'missing_modules': MISSING_MODULES,
            'ai_metrics': {
                'requests_processed': self.ai_stats['requests_processed'],
                'successful_requests': self.ai_stats['successful_requests'],
                'failed_requests': self.ai_stats['failed_requests'],
                'success_rate': (self.ai_stats['successful_requests'] / max(1, self.ai_stats['requests_processed'])) * 100,
                'avg_response_time': self.ai_stats['avg_response_time'],
                'spatial_enhancements': self.ai_stats['spatial_enhancements']
            },
            'system_metrics': {
                'uptime_hours': uptime_seconds / 3600,
                'active_processes': self.system_stats['active_processes'],
                'error_count': self.system_stats['error_count']
            },
            'subsystem_status': {
                'core_system': self.llammy_core is not None,
                'spatial_intelligence': self.spatial_intelligence is not None,
                'rag_system': self.rag_system is not None,
                'dual_ai': self.dual_ai_engine is not None
            }
        }

# Global integrated orchestrator
integrated_orchestrator = IntegratedModuleOrchestrator()

# =============================================================================
# UI PANELS
# =============================================================================

class MCP_PT_MainPanel(bpy.types.Panel):
    """Creates a Panel in the 3D Viewport Sidebar"""
    bl_label = "Llammy MCP AI"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Llammy"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.llammy_mcp_props
        
        # Connection Status
        box = layout.box()
        box.label(text="MCP Bridge Status", icon='NETWORK_VEC')
        if props.is_connected:
            box.label(text=f"Status: {props.status_message}", icon='PLUG')
            box.label(text=f"Last Heartbeat: {props.last_heartbeat}")
        else:
            box.label(text=f"Status: {props.status_message}", icon='UNPLUGGED')
        
        layout.separator()
        
        # Connection Operators
        row = layout.row(align=True)
        row.operator("mcp.connect_server", text="Connect", icon='LINK_URL')
        row.operator("mcp.disconnect_server", text="Disconnect", icon='UNLINK')
        
        layout.separator()
        
        # Debug Logs
        box = layout.box()
        box.label(text="Debug Log", icon='INFO')
        for log in reversed(debug.logs):
            box.label(text=log)
        # Spatial Intelligence status
        if init_status.get('spatial_intelligence_active'):
            status_col.label(text="✓ Spatial AI: ACTIVE", icon='ORIENTATION_VIEW')
        else:
            status_col.label(text="✗ Spatial AI: INACTIVE", icon='CANCEL')
        
        # RAG Business status
        if init_status.get('rag_business_active'):
            status_col.label(text="✓ RAG System: ACTIVE", icon='FILE_TEXT')
        else:
            status_col.label(text="✗ RAG System: INACTIVE", icon='CANCEL')
        
        # Dual AI status
        if init_status.get('dual_ai_active'):
            status_col.label(text="✓ Dual AI: ACTIVE", icon='LIGHTPROBE_VOLUME')
        else:
            status_col.label(text="✗ Dual AI: INACTIVE", icon='CANCEL')
        
        # Model connection status
        model_status = model_manager.get_connection_status()
        if model_status['status'] == 'connected':
            status_col.label(text=f"✓ Models: {model_status['cached_models']}", icon='NETWORK_DRIVE')
        else:
            status_col.label(text=f"✗ Models: {model_status['status']}", icon='CANCEL')

class LLAMMY_PT_RequestPanel(bpy.types.Panel):
    bl_label = "AI Request"
    bl_idname = "LLAMMY_PT_request_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    bl_parent_id = "LLAMMY_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Dual Model Configuration - Only 2 models needed
        model_box = layout.box()
        model_box.label(text="Dual AI Configuration", icon='SETTINGS')
        
        # Creative Model (Analysis/Planning)
        if hasattr(scene, "llammy_creative_model"):
            creative_row = model_box.row()
            creative_row.label(text="Creative:", icon='COLOR')
            creative_row.prop(scene, "llammy_creative_model", text="")
        
        # Technical Model (Implementation/Code)
        if hasattr(scene, "llammy_technical_model"):
            technical_row = model_box.row()
            technical_row.label(text="Technical:", icon='TOOL_SETTINGS')
            technical_row.prop(scene, "llammy_technical_model", text="")
        
        # Model status and refresh
        model_status = model_manager.get_connection_status()
        status_row = model_box.row()
        if model_status['status'] == 'connected':
            status_row.label(text=f"✓ {model_status['cached_models']} models", icon='NETWORK_DRIVE')
        else:
            status_row.label(text=f"✗ {model_status['status']}", icon='CANCEL')
        
        status_row.operator("llammy.refresh_models", text="", icon='FILE_REFRESH')
        
        # Enhanced prompt input section
        prompt_box = layout.box()
        prompt_box.label(text="AI Prompt (Spatial Intelligence Enabled)", icon='CONSOLE')
        
        # Main prompt input
        if hasattr(scene, "llammy_request_input"):
            col = prompt_box.column(align=True)
            col.prop(scene, "llammy_request_input", text="")
            
            # Prompt character count
            prompt_text = getattr(scene, 'llammy_request_input', '')
            char_count = len(prompt_text)
            char_row = prompt_box.row()
            if char_count > 1500:
                char_row.label(text=f"Characters: {char_count}/2000", icon='ERROR')
            elif char_count > 1000:
                char_row.label(text=f"Characters: {char_count}/2000", icon='INFO')
            else:
                char_row.label(text=f"Characters: {char_count}/2000")
        
        # Quick prompt templates
        templates_row = prompt_box.row()
        anim_op = templates_row.operator("llammy.load_template", text="Animation")
        anim_op.template_type = "animation"
        debug_op = templates_row.operator("llammy.load_template", text="Debug")
        debug_op.template_type = "debug"
        spatial_op = templates_row.operator("llammy.load_template", text="Spatial")
        spatial_op.template_type = "spatial"
        
        # Context information
        file_summary = file_manager.get_summary()
        if file_summary['total_files'] > 0:
            context_box = prompt_box.box()
            context_row = context_box.row()
            context_row.label(text="Context Files:", icon='FILE_TEXT')
            context_row.label(text=f"{file_summary['total_files']} files ({file_summary['total_size_mb']:.1f}MB)")
            
            # Show file types
            if file_summary['file_types']:
                types_text = ", ".join([f"{ft}: {ct}" for ft, ct in list(file_summary['file_types'].items())[:3]])
                context_box.label(text=types_text)
        
        # Execute button with enhanced styling
        execute_section = layout.column()
        execute_section.separator()
        execute_row = execute_section.row()
        execute_row.scale_y = 1.8
        
        # Show different button text based on available systems
        status = integrated_orchestrator.get_comprehensive_status()
        spatial_active = status['initialization_status'].get('spatial_intelligence_active', False)
        
        if file_summary['total_files'] > 0 and spatial_active:
            execute_row.operator("llammy.execute_integrated_request", text="🚀 EXECUTE WITH SPATIAL AI + RAG", icon='PLAY')
        elif spatial_active:
            execute_row.operator("llammy.execute_integrated_request", text="🚀 EXECUTE WITH SPATIAL AI", icon='PLAY')
        elif file_summary['total_files'] > 0:
            execute_row.operator("llammy.execute_integrated_request", text="🚀 EXECUTE WITH RAG", icon='PLAY')
        else:
            execute_row.operator("llammy.execute_integrated_request", text="🚀 EXECUTE REQUEST", icon='PLAY')

class LLAMMY_PT_FilesPanel(bpy.types.Panel):
    bl_label = "Context Files"
    bl_idname = "LLAMMY_PT_files_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    bl_parent_id = "LLAMMY_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        
        file_summary = file_manager.get_summary()
        
        # File statistics
        if file_summary['total_files'] > 0:
            stats_box = layout.box()
            stats_box.label(text="File Statistics", icon='FILE_TEXT')
            
            stats_col = stats_box.column(align=True)
            stats_col.label(text=f"Files: {file_summary['total_files']}")
            stats_col.label(text=f"Size: {file_summary['total_size_mb']:.1f}MB")
            
            if file_summary.get('largest_file'):
                stats_col.label(text=f"Largest: {file_summary['largest_file']}")
            
            # File types
            if file_summary['file_types']:
                for file_type, count in list(file_summary['file_types'].items())[:3]:
                    stats_col.label(text=f"{file_type}: {count}")
        
        # File operations
        ops_row = layout.row()
        ops_row.operator("llammy.select_files", text="Add Files", icon='FILEBROWSER')
        ops_row.operator("llammy.select_folder", text="Add Folder", icon='FILE_FOLDER')
        
        if file_summary['total_files'] > 0:
            clear_row = layout.row()
            clear_row.operator("llammy.clear_files", text="Clear All", icon='TRASH')

class LLAMMY_PT_SpatialPanel(bpy.types.Panel):
    bl_label = "Spatial Intelligence"
    bl_idname = "LLAMMY_PT_spatial_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    bl_parent_id = "LLAMMY_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        
        # Spatial Intelligence Status
        status = integrated_orchestrator.get_comprehensive_status()
        spatial_active = status['initialization_status'].get('spatial_intelligence_active', False)
        
        if spatial_active:
            # Scene analysis info
            if integrated_orchestrator.spatial_intelligence:
                try:
                    spatial_status = integrated_orchestrator.spatial_intelligence.get_module_status()
                    scene_graph_status = spatial_status.get('scene_graph_status', {})
                    
                    info_box = layout.box()
                    info_box.label(text="Scene Analysis", icon='ORIENTATION_VIEW')
                    
                    info_col = info_box.column(align=True)
                    info_col.label(text=f"Objects: {scene_graph_status.get('total_nodes', 0)}")
                    info_col.label(text=f"Relationships: {scene_graph_status.get('total_relationships', 0)}")
                    info_col.label(text=f"Complexity: {scene_graph_status.get('complexity_score', 0):.1f}")
                    
                    # Semantic groups
                    semantic_groups = scene_graph_status.get('semantic_groups', {})
                    if semantic_groups:
                        info_col.separator()
                        info_col.label(text="Semantic Groups:")
                        for group, count in list(semantic_groups.items())[:3]:
                            if count > 0:
                                info_col.label(text=f"  {group}: {count}")
                
                except Exception as e:
                    layout.label(text=f"Status error: {str(e)[:30]}...")
            
            # Refresh scene graph button
            refresh_row = layout.row()
            refresh_row.operator("llammy.refresh_spatial", text="Refresh Scene Graph", icon='FILE_REFRESH')
            
            # Statistics
            if integrated_orchestrator.spatial_intelligence:
                stats = integrated_orchestrator.spatial_intelligence.stats
                stats_box = layout.box()
                stats_box.label(text="Spatial Statistics", icon='INFO')
                
                stats_col = stats_box.column(align=True)
                stats_col.label(text=f"Scenes Analyzed: {stats.get('scenes_analyzed', 0)}")
                stats_col.label(text=f"Queries Processed: {stats.get('queries_processed', 0)}")
                stats_col.label(text=f"AI Integrations: {stats.get('ai_integrations', 0)}")
        else:
            # Not active - show initialization option
            warning_box = layout.box()
            warning_box.label(text="Spatial Intelligence Inactive", icon='ERROR')
            
            if not MODULES_LOADED:
                warning_box.label(text="Missing modules detected")
                for module in MISSING_MODULES[:2]:  # Show first 2 errors
                    warning_box.label(text=f"  {module[:40]}...")
            else:
                warning_box.label(text="Scene analysis not available")
                warning_box.operator("llammy.initialize_systems", text="Initialize", icon='PLAY')

class LLAMMY_PT_MetricsPanel(bpy.types.Panel):
    bl_label = "Performance Metrics"
    bl_idname = "LLAMMY_PT_metrics_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    bl_parent_id = "LLAMMY_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        
        status = integrated_orchestrator.get_comprehensive_status()
        
        # AI Metrics
        if status.get('ai_metrics'):
            ai_box = layout.box()
            ai_box.label(text="AI Metrics", icon='SHADERFX')
            
            ai_metrics = status['ai_metrics']
            ai_col = ai_box.column(align=True)
            ai_col.label(text=f"Requests: {ai_metrics['requests_processed']}")
            ai_col.label(text=f"Success: {ai_metrics['successful_requests']}")
            ai_col.label(text=f"Failed: {ai_metrics['failed_requests']}")
            ai_col.label(text=f"Success Rate: {ai_metrics['success_rate']:.1f}%")
            
            if ai_metrics['avg_response_time'] > 0:
                ai_col.label(text=f"Avg Response: {ai_metrics['avg_response_time']:.2f}s")
            
            if ai_metrics['spatial_enhancements'] > 0:
                ai_col.label(text=f"Spatial Enhanced: {ai_metrics['spatial_enhancements']}")
        
        # System Metrics
        if status.get('system_metrics'):
            system_box = layout.box()
            system_box.label(text="System Metrics", icon='SETTINGS')
            
            system_metrics = status['system_metrics']
            system_col = system_box.column(align=True)
            system_col.label(text=f"Uptime: {system_metrics['uptime_hours']:.1f}h")
            system_col.label(text=f"Active Modules: {system_metrics['active_processes']}")
            system_col.label(text=f"Errors: {system_metrics['error_count']}")
        
        # Subsystem Status
        if status.get('subsystem_status'):
            subsystem_box = layout.box()
            subsystem_box.label(text="Subsystems", icon='MODIFIER')
            
            subsystem_status = status['subsystem_status']
            subsystem_col = subsystem_box.column(align=True)
            
            for system, active in subsystem_status.items():
                icon = 'CHECKMARK' if active else 'CANCEL'
                status_text = 'ACTIVE' if active else 'INACTIVE'
                subsystem_col.label(text=f"{system}: {status_text}", icon=icon)

class LLAMMY_PT_DebugPanel(bpy.types.Panel):
    bl_label = "Debug & Logs"
    bl_idname = "LLAMMY_PT_debug_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Llammy'
    bl_parent_id = "LLAMMY_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        
        # Recent logs
        logs_box = layout.box()
        logs_box.label(text="Recent Logs", icon='TEXT')
        
        logs = debug.get_recent_logs(5)
        for log in logs:
            # Truncate long logs and remove timestamps for UI
            display_log = log.split('] ', 2)[-1] if '] ' in log else log
            display_log = display_log[:50] + "..." if len(display_log) > 50 else display_log
            logs_box.label(text=display_log)
        
        # Debug operations
        debug_row = layout.row()
        debug_row.operator("llammy.test_connection", text="Test Connection", icon='PLUGIN')
        debug_row.operator("llammy.clear_logs", text="Clear", icon='TRASH')

# =============================================================================
# OPERATORS
# =============================================================================

class MCP_OT_ConnectServer(bpy.types.Operator):
    """Connect to the MCP core server"""
    bl_idname = "mcp.connect_server"
    bl_label = "Connect MCP Server"
    
    def execute(self, context):
        thread = threading.Thread(target=asyncio.run, args=(mcp_bridge.connect(),))
        thread.daemon = True
        thread.start()
        return {'FINISHED'}

class MCP_OT_DisconnectServer(bpy.types.Operator):
    """Disconnect from the MCP core server"""
    bl_idname = "mcp.disconnect_server"
    bl_label = "Disconnect MCP Server"
    
    def execute(self, context):
        thread = threading.Thread(target=asyncio.run, args=(mcp_bridge.disconnect(),))
        thread.daemon = True
        thread.start()
        return {'FINISHED'}
class LLAMMY_OT_ExecuteIntegratedRequest(bpy.types.Operator):
    bl_idname = "llammy.execute_integrated_request"
    bl_label = "Execute Integrated AI Request"
    bl_description = "Execute AI request through integrated pipeline with spatial intelligence"
    
    def execute(self, context):
        scene = context.scene
        request = getattr(scene, 'llammy_request_input', '')
        
        if not request:
            self.report({'ERROR'}, "No request entered")
            return {'CANCELLED'}
        
        start_time = time.time()
        
        try:
            # Get selected models
            creative_model = getattr(scene, 'llammy_creative_model', 'default')
            technical_model = getattr(scene, 'llammy_technical_model', 'default')
            
            debug.log("INFO", f"Processing integrated request: {request[:50]}...")
            debug.log("INFO", f"Using Creative: {creative_model}, Technical: {technical_model}")
            
            # Get context files
            context_files = file_manager.get_files_for_context()
            
            # Execute through integrated pipeline
            result = integrated_orchestrator.execute_integrated_request(
                request,
                creative_model,
                technical_model,
                context_files
            )
            
            processing_time = time.time() - start_time
            
            if result.get('success'):
                # Success metrics
                enhancements = result.get('enhancements_used', {})
                enhancement_text = []
                if enhancements.get('spatial_intelligence'):
                    enhancement_text.append("Spatial AI")
                if enhancements.get('rag_context'):
                    enhancement_text.append("RAG Context")
                if enhancements.get('dual_ai'):
                    enhancement_text.append("Dual AI")
                
                enhancement_str = " + ".join(enhancement_text) if enhancement_text else "Basic"
                
                self.report({'INFO'}, f"Request processed successfully in {processing_time:.2f}s with {enhancement_str}!")
                
                # Log success details
                debug.log("INFO", f"Success: {len(result.get('generated_code', ''))} chars generated")
                if result.get('spatial_context', {}).get('relevant_objects'):
                    debug.log("INFO", f"Spatial context: {len(result['spatial_context']['relevant_objects'])} relevant objects")
                
            else:
                error = result.get('error', 'Unknown error')
                self.report({'ERROR'}, f"Request failed: {error}")
                debug.log("ERROR", f"Request failed: {error}")
        
        except Exception as e:
            processing_time = time.time() - start_time
            debug.log("ERROR", f"Request execution failed: {e}")
            self.report({'ERROR'}, f"Request failed: {e}")
        
        return {'FINISHED'}

class LLAMMY_OT_LoadTemplate(bpy.types.Operator):
    bl_idname = "llammy.load_template"
    bl_label = "Load Template"
    bl_description = "Load a predefined prompt template"
    
    template_type: bpy.props.StringProperty()
    
    def execute(self, context):
        scene = context.scene
        
        templates = {
            "animation": "Create a keyframe animation for the selected object. Focus on smooth transitions and realistic timing. Consider easing and natural motion principles.",
            "debug": "Analyze the current scene for potential issues. Check for: missing materials, unused objects, optimization opportunities, and structural problems.",
            "spatial": "Create objects with spatial awareness. Consider existing object positions, avoid overlaps, and maintain scene composition. Use spatial relationships effectively."
        }
        
        if self.template_type in templates:
            scene.llammy_request_input = templates[self.template_type]
            self.report({'INFO'}, f"Loaded {self.template_type} template")
        else:
            self.report({'ERROR'}, "Unknown template type")
        
        return {'FINISHED'}

class LLAMMY_OT_RefreshSpatial(bpy.types.Operator):
    bl_idname = "llammy.refresh_spatial"
    bl_label = "Refresh Spatial Intelligence"
    bl_description = "Refresh scene graph analysis"
    
    def execute(self, context):
        if integrated_orchestrator.spatial_intelligence:
            try:
                success = integrated_orchestrator.spatial_intelligence.refresh_scene_graph()
                if success:
                    self.report({'INFO'}, "Scene graph refreshed successfully")
                else:
                    self.report({'WARNING'}, "Scene graph refresh failed")
            except Exception as e:
                self.report({'ERROR'}, f"Refresh failed: {e}")
        else:
            self.report({'ERROR'}, "Spatial Intelligence not available")
        
        return {'FINISHED'}

class LLAMMY_OT_InitializeSystems(bpy.types.Operator):
    bl_idname = "llammy.initialize_systems"
    bl_label = "Initialize Systems"
    bl_description = "Initialize all Llammy systems"
    
    def execute(self, context):
        try:
            success = integrated_orchestrator.initialize()
            if success:
                self.report({'INFO'}, "Systems initialized successfully")
            else:
                self.report({'WARNING'}, "System initialization completed with warnings")
        except Exception as e:
            self.report({'ERROR'}, f"Initialization failed: {e}")
        
        return {'FINISHED'}

class LLAMMY_OT_SelectFiles(bpy.types.Operator):
    bl_idname = "llammy.select_files"
    bl_label = "Select Files"
    bl_description = "Select files for AI context"
    
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement)
    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    
    def execute(self, context):
        file_paths = []
        for file_elem in self.files:
            file_path = os.path.join(self.directory, file_elem.name)
            file_paths.append(file_path)
        
        file_manager.add_files(file_paths)
        summary = file_manager.get_summary()
        self.report({'INFO'}, f"Added {len(file_paths)} files. Total: {summary['total_files']}")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class LLAMMY_OT_SelectFolder(bpy.types.Operator):
    bl_idname = "llammy.select_folder"
    bl_label = "Select Folder"
    bl_description = "Select a folder to add all supported files"
    
    directory: bpy.props.StringProperty(subtype='DIR_PATH')
    
    def execute(self, context):
        if self.directory and os.path.isdir(self.directory):
            file_manager.add_files([self.directory])
            summary = file_manager.get_summary()
            self.report({'INFO'}, f"Added folder: {summary['total_files']} total files")
        else:
            self.report({'ERROR'}, "Invalid directory selected")
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class LLAMMY_OT_ClearFiles(bpy.types.Operator):
    bl_idname = "llammy.clear_files"
    bl_label = "Clear Files"
    bl_description = "Clear all selected files"
    
    def execute(self, context):
        file_manager.clear_files()
        self.report({'INFO'}, "All files cleared")
        return {'FINISHED'}

class LLAMMY_OT_RefreshModels(bpy.types.Operator):
    bl_idname = "llammy.refresh_models"
    bl_label = "Refresh Models"
    bl_description = "Refresh available Ollama models"
    
    def execute(self, context):
        models = model_manager.refresh_models()
        connection_status = model_manager.get_connection_status()
        
        if connection_status['status'] == 'connected':
            self.report({'INFO'}, f"Found {len(models)} models")
        else:
            self.report({'WARNING'}, f"Connection issue: {connection_status['status']}")
        
        return {'FINISHED'}

class LLAMMY_OT_TestConnection(bpy.types.Operator):
    bl_idname = "llammy.test_connection"
    bl_label = "Test Connection"
    bl_description = "Test all system connections"
    
    def execute(self, context):
        # Test Ollama connection
        model_status = model_manager.get_connection_status()
        
        # Test integrated systems
        status = integrated_orchestrator.get_comprehensive_status()
        
        messages = []
        
        if model_status['status'] == 'connected':
            messages.append(f"✓ Ollama: {model_status['cached_models']} models")
        else:
            messages.append(f"✗ Ollama: {model_status['status']}")
        
        if status.get('initialization_status', {}).get('core_initialized'):
            messages.append("✓ Core: Ready")
        else:
            messages.append("✗ Core: Not initialized")
        
        if status.get('initialization_status', {}).get('spatial_intelligence_active'):
            messages.append("✓ Spatial AI: Active")
        else:
            messages.append("✗ Spatial AI: Inactive")
        
        # Show results
        for msg in messages:
            if "✓" in msg:
                self.report({'INFO'}, msg)
            else:
                self.report({'WARNING'}, msg)
        
        return {'FINISHED'}

class LLAMMY_OT_ClearLogs(bpy.types.Operator):
    bl_idname = "llammy.clear_logs"
    bl_label = "Clear Logs"
    bl_description = "Clear debug logs"
    
    def execute(self, context):
        debug.clear_logs()
        self.report({'INFO'}, "Logs cleared")
        return {'FINISHED'}

# =============================================================================
# PROPERTIES - ONLY 2 MODEL SELECTORS NEEDED
# =============================================================================

# Add all classes that should be registered
classes = [
    MCP_PT_MainPanel,
    MCP_OT_ConnectServer,
    MCP_OT_DisconnectServer,
    LlammyMCPProperties,
]

def register():
    """Register all classes and properties"""
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.llammy_mcp_props = bpy.props.PointerProperty(type=LlammyMCPProperties)
    
    # Run a simple timer to process the queue from the main thread
    bpy.app.timers.register(mcp_bridge.process_queue, persistent=True)
    
    debug.log("INFO", "🚀 Llammy MCP AI Bridge addon registered")
def register_properties():
    """Register all Blender properties"""
    
    # Request input with larger character limit
    bpy.types.Scene.llammy_request_input = bpy.props.StringProperty(
        name="AI Request",
        description="Your request for the integrated AI system with spatial intelligence",
        default="",
        maxlen=2000
    )
    
    # Creative model (Analysis/Planning phase)
    bpy.types.Scene.llammy_creative_model = bpy.props.EnumProperty(
        name="Creative Model",
        description="Model for creative analysis and planning phase",
        items=get_models_for_properties,
        default=0
    )
    
    # Technical model (Implementation/Code phase)
    bpy.types.Scene.llammy_technical_model = bpy.props.EnumProperty(
        name="Technical Model",
        description="Model for technical implementation and code generation phase",
        items=get_models_for_properties,
        default=0
    )

def unregister():
    """Unregister all classes and properties"""
    # Clean up the timer
    bpy.app.timers.unregister(mcp_bridge.process_queue)
    
    # Clean up the connection when the addon is disabled
    if mcp_bridge.connected:
        try:
            thread = threading.Thread(target=asyncio.run, args=(mcp_bridge.disconnect(),))
            thread.daemon = True
            thread.start()
            thread.join(timeout=2) # Give it a moment to close
        except Exception as e:
            debug.log("ERROR", f"Failed to gracefully unregister: {e}")
            
    # Unregister properties
    del bpy.types.Scene.llammy_mcp_props

    # Unregister classes in reverse order
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    debug.log("INFO", "👋 Llammy MCP AI Bridge addon unregistered")
def unregister_properties():
    """Unregister all properties"""
    props_to_remove = [
        'llammy_request_input',
        'llammy_creative_model',
        'llammy_technical_model'
    ]
    
    for prop in props_to_remove:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

# =============================================================================
# CLASSES LIST
# =============================================================================

classes = [
    LLAMMY_PT_MainPanel,
    LLAMMY_PT_RequestPanel,
    LLAMMY_PT_FilesPanel,
    LLAMMY_PT_SpatialPanel,
    LLAMMY_PT_MetricsPanel,
    LLAMMY_PT_DebugPanel,
    LLAMMY_OT_ExecuteIntegratedRequest,
    LLAMMY_OT_LoadTemplate,
    LLAMMY_OT_RefreshSpatial,
    LLAMMY_OT_InitializeSystems,
    LLAMMY_OT_SelectFiles,
    LLAMMY_OT_SelectFolder,
    LLAMMY_OT_ClearFiles,
    LLAMMY_OT_RefreshModels,
    LLAMMY_OT_TestConnection,
    LLAMMY_OT_ClearLogs,
]

# =============================================================================
# REGISTRATION
# =============================================================================

def register():
    """Register the addon"""
    
    # Initialize managers
    debug.log("INFO", "Starting Llammy Spatial AI initialization...")
    
    # Register classes
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
            debug.log("DEBUG", f"Registered class: {cls.__name__}")
        except Exception as e:
            debug.log("ERROR", f"Failed to register {cls.__name__}: {e}")
    
    # Register properties
    register_properties()
    debug.log("INFO", "Properties registered")
    
    # Initialize integrated systems
    if integrated_orchestrator.initialize():
        debug.log("INFO", "Integrated systems initialized successfully")
    else:
        debug.log("ERROR", "Integrated systems initialization failed")
    
    # Initialize model manager
    models = model_manager.get_available_models()
    debug.log("INFO", f"Model manager initialized with {len(models)} models")
    
    # Final status report
    status = integrated_orchestrator.get_comprehensive_status()
    debug.log("INFO", "=== LLAMMY SPATIAL AI READY ===")
    debug.log("INFO", f"Core Status: {'ACTIVE' if status['initialization_status']['core_initialized'] else 'INACTIVE'}")
    debug.log("INFO", f"Spatial Intelligence: {'ACTIVE' if status['initialization_status']['spatial_intelligence_active'] else 'INACTIVE'}")
    debug.log("INFO", f"RAG Business: {'ACTIVE' if status['initialization_status']['rag_business_active'] else 'INACTIVE'}")
    debug.log("INFO", f"Dual AI: {'ACTIVE' if status['initialization_status']['dual_ai_active'] else 'INACTIVE'}")
    debug.log("INFO", f"Dependencies: {'OK' if status['initialization_status']['dependencies_ok'] else 'MISSING'}")
    debug.log("INFO", "===============================")
    
    print("Llammy Spatial AI v6.0 - Addon registered successfully!")
    print("Enhanced Features:")
    print("  ✓ Integrated dual AI architecture with spatial intelligence")
    print("  ✓ 3D scene graph generation and spatial reasoning")
    print("  ✓ RAG business system with embedded model support")
    print("  ✓ Self-debugging and adaptive learning")
    print("  ✓ Swappable Ollama model support with caching")
    print("  ✓ Comprehensive performance metrics")
    print("  ✓ Real-time spatial relationship analysis")

def unregister():
    """Unregister the addon"""
    
    debug.log("INFO", "Unregistering Llammy Spatial AI...")
    
    # Unregister properties
    unregister_properties()
    
    # Unregister classes in reverse order
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception as e:
            debug.log("WARNING", f"Failed to unregister {cls.__name__}: {e}")
    
    debug.log("INFO", "Llammy Spatial AI unregistered")
    print("Llammy Spatial AI - Addon unregistered")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    register()

# Print statements for the Blender console, confirming module load
print("LLAMMY MCP BLENDER ADDON LOADED!")
print("This version is re-engineered for the MCP architecture.")
print("It uses a WebSocket bridge to connect to the external core server.")
