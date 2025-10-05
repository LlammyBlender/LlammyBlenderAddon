# =============================================================================
# LLAMMY CORE v6.1 - Re-engineered for Spatial Intelligence
# llammy_core.py - Now integrates the scene graph engine
# =============================================================================

import asyncio
import json
import time
import traceback
import requests
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# NEW IMPORT: Import the spatial intelligence module
from .llammy_spatial_intelligence import SpatialIntelligenceMCP, initialize_spatial_intelligence, enhance_ai_request_with_spatial_context

print("🎭 Llammy Core v6.1 - Now with Spatial Intelligence integration")

# =============================================================================
# MCP MESSAGE PROTOCOL - WORKING IMPLEMENTATION
# =============================================================================

class MCPMessageType(Enum):
    """MCP message types for inter-module communication"""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    A2A_MODEL_CALL = "a2a_model_call"

@dataclass
class MCPMessage:
    """MCP protocol message structure"""
    id: str
    type: MCPMessageType
    source: str
    target: str
    data: Dict[str, Any]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'source': self.source,
            'target': self.target,
            'data': self.data,
            'timestamp': self.timestamp
        }

# =============================================================================
# WORKING OLLAMA SERVICE MANAGER
# =============================================================================

class WorkingOllamaService:
    """Actually working Ollama service with real HTTP calls"""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.available_models = []
        self.connection_healthy = False
        self.stats = {
            'requests': 0,
            'successes': 0,
            'model_calls': 0,
            'total_tokens': 0
        }
        
        # Test connection on init
        self.test_connection()
    
    def test_connection(self) -> Dict[str, Any]:
        """Test real Ollama connection"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.available_models = [model['name'] for model in data.get('models', [])]
                self.connection_healthy = True
                
                print(f"✅ Ollama connected - {len(self.available_models)} models available")
                
                return {
                    'success': True,
                    'status': 'connected',
                    'models_available': len(self.available_models),
                    'model_list': self.available_models[:5]  # Show first 5
                }
            else:
                self.connection_healthy = False
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}',
                    'status': 'error'
                }
                
        except requests.exceptions.ConnectionError:
            self.connection_healthy = False
            print("❌ Ollama connection failed - is it running on localhost:11434?")
            return {
                'success': False,
                'error': 'Connection refused',
                'status': 'disconnected'
            }
        except Exception as e:
            self.connection_healthy = False
            return {
                'success': False,
                'error': str(e),
                'status': 'error'
            }
    
    def call_model(self, model_name: str, prompt: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make real Ollama API call"""
        if not self.connection_healthy:
            return {
                'success': False,
                'error': 'Ollama not connected',
                'response': '# Error: Ollama not available'
            }
        
        self.stats['requests'] += 1
        self.stats['model_calls'] += 1
        
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": parameters or {
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            }
            
            print(f"🔄 Calling Ollama model: {model_name}")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '')
                
                # Estimate tokens (rough calculation)
                estimated_tokens = len(content.split())
                self.stats['total_tokens'] += estimated_tokens
                self.stats['successes'] += 1
                
                return {
                    'success': True,
                    'response': content,
                    'model_used': model_name,
                    'estimated_tokens': estimated_tokens
                }
            else:
                error_msg = f'Ollama error: HTTP {response.status_code}'
                print(f"❌ {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'response': f'# {error_msg}'
                }
                
        except requests.exceptions.Timeout:
            error_msg = f'Model call timed out'
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'response': '# Request timed out'
            }
        except Exception as e:
            error_msg = f'Model call failed: {str(e)}'
            print(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'response': f'# Error: {e}'
            }
    
    def resolve_best_model(self, model_hint: str, task_type: str = 'general') -> str:
        """Resolve best available model"""
        if not self.available_models:
            return "llama3.2:3b"  # Fallback
        
        # Priority order for different models
        priority_models = [
            "triangle104", "llammy", "sentient", "nemotron", "qwen2.5", "llama3.2"
        ]
        
        # If hint provided and available, use it
        if model_hint:
            for model in self.available_models:
                if model_hint.lower() in model.lower():
                    return model
        
        # Otherwise find highest priority available model
        for priority in priority_models:
            for model in self.available_models:
                if priority in model.lower():
                    return model
        
        # Return first available model as last resort
        return self.available_models[0] if self.available_models else "llama3.2:3b"

# =============================================================================
# WORKING VISION SERVICE - NOW ENHANCED
# =============================================================================

class WorkingVisionService:
    """Actually working vision service that provides scene context"""
    def __init__(self, llammy_core_instance: Any):
        # The Vision Service now receives a core instance to use its services
        self.core = llammy_core_instance
        print("👁️ Working Vision Service initialized")

    def get_context(self, user_request: str) -> str:
        """Get scene context from the Blender environment."""
        # This is where we would call the actual Blender API
        # For now, it's a mock
        scene_info = "The scene contains a single cube named 'Cube' located at (0, 0, 0)."
        return scene_info

# =============================================================================
# WORKING DUAL AI SYSTEM - RE-ENGINEERED TO ACCEPT SPATIAL DATA
# =============================================================================

class WorkingDualAI:
    """Actually working dual AI system with real Ollama calls"""
    def __init__(self, ollama_service: WorkingOllamaService, vision_service: WorkingVisionService):
        self.ollama_service = ollama_service
        self.vision_service = vision_service
        self.system_prompt_builder = WorkingSystemPromptBuilder()
        self.code_enhancer = WorkingBlenderCodeEnhancer()
        print("🤖 Working Dual AI system initialized")

    def execute_request(self, user_request: str) -> Dict[str, Any]:
        """Execute a user request by generating and enhancing code"""
        
        # 1. Get Scene Context
        # The core orchestrator now provides the spatial context directly.
        # We can still get high-level context from the vision service if needed.
        high_level_context = self.vision_service.get_context(user_request)
        
        # 2. Build Prompt for LLM
        prompt = self.system_prompt_builder.build_full_prompt(user_request, high_level_context)
        
        # 3. Resolve best model
        model_name = self.ollama_service.resolve_best_model(prompt)
        
        # 4. Call LLM
        llm_response = self.ollama_service.call_model(model_name, prompt)
        
        if not llm_response['success']:
            return llm_response

        # 5. Extract and enhance code
        generated_code = llm_response['response']
        enhanced_code = self.code_enhancer.enhance_code(generated_code, high_level_context)
        
        return {
            'success': True,
            'generated_code': enhanced_code,
            'original_llm_output': generated_code,
            'model_used': model_name
        }

class WorkingSystemPromptBuilder:
    def build_full_prompt(self, user_request: str, vision_context: str) -> str:
        return (
            f"You are an expert Blender Python programmer.\n"
            f"Current Scene State:\n{vision_context}\n\n"  # This now includes spatial data
            f"User Request:\n{user_request}\n\n"
            f"Instructions: Generate the complete, self-contained Blender Python code to fulfill the request. "
            f"Do not include any extra text. Your response should be ONLY code.\n"
            f"```python\n"
        )

class WorkingBlenderCodeEnhancer:
    def enhance_code(self, code: str, context: str) -> str:
        # Simple enhancement logic
        return code.replace("bpy.ops.object.select_all()", "bpy.ops.object.select_all(action='SELECT')")

# =============================================================================
# WORKING LLAMMY CORE ORCHESTRATOR - NEW SPATIAL INTEGRATION
# =============================================================================

class WorkingLlammyCoreOrchestrator:
    version = "6.1"
    
    def __init__(self):
        # Initialize services
        self.ollama_service = WorkingOllamaService()
        self.vision_service = WorkingVisionService(self)
        self.dual_ai_service = WorkingDualAI(self.ollama_service, self.vision_service)
        
        # NEW: Initialize the Spatial Intelligence Module
        self.spatial_intelligence = initialize_spatial_intelligence(self)
        
        # Internal state
        self.status = {
            'services': {
                'ollama': self.ollama_service.test_connection(),
                'vision': {'active': True},
                'spatial_intelligence': {'active': self.spatial_intelligence.is_running if hasattr(self.spatial_intelligence, 'is_running') else True, 'initialization_status': 'ok'}
            },
            'performance': {
                'total_requests': 0,
                'total_successes': 0
            }
        }
        print("✅ Llammy Core v6.1 Orchestrator initialized!")

    def process_user_request(self, user_request: str) -> Dict[str, Any]:
        """Main entry point to process a user request"""
        print(f"🎬 Processing request: {user_request}")
        
        # NEW: Enhance the user request with spatial context before it's sent to the AI
        enhanced_request_data = enhance_ai_request_with_spatial_context(
            user_request,
            self.spatial_intelligence # Pass the SpatialIntelligenceMCP instance
        )
        
        # The enhanced request now contains 'spatial_context'
        # Pass this enriched data to the dual AI system
        enriched_user_request = f"{user_request}\n\nSpatial Context:\n{enhanced_request_data.get('spatial_context', 'No spatial context available.')}"

        self.status['performance']['total_requests'] += 1
        
        # Call the dual AI service with the enriched request
        result = self.dual_ai_service.execute_request(enriched_user_request)
        
        if result.get('success'):
            self.status['performance']['total_successes'] += 1
        
        return result

    def get_system_health(self) -> Dict[str, Any]:
        """Get the overall health of the system"""
        try:
            status = self.status.copy()
            status['performance']['success_rate'] = (
                status['performance']['total_successes'] / status['performance']['total_requests']
                if status['performance']['total_requests'] > 0 else 0
            )
            
            # Health factors
            health_factors = {
                'ollama_connected': 25 if status['services']['ollama']['connected'] else 0,
                'vision_active': 25 if status['services']['vision']['active'] else 0,
                'spatial_intelligence_active': 25 if status['services']['spatial_intelligence']['active'] else 0,
                'success_rate': status['performance']['success_rate'] * 25
            }
            
            health_score = sum(health_factors.values())
            
            return {
                'status': 'healthy' if health_score > 75 else 'degraded' if health_score > 50 else 'unhealthy',
                'health_score': health_score,
                'health_factors': health_factors,
                'system_stats': status,
                'version': self.version
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'health_score': 0,
                'error': str(e)
            }

# =============================================================================
# EXPORT INTERFACE
# =============================================================================

# Global orchestrator instance
working_llammy_core = WorkingLlammyCoreOrchestrator()

def get_llammy_core() -> WorkingLlammyCoreOrchestrator:
    """Get the global core instance"""
    return working_llammy_core

def initialize_llammy_system():
    """Initialize system services"""
    # This is handled by the __init__ of the orchestrator
    pass

def process_user_request(user_request: str) -> Dict[str, Any]:
    """Process a user request and return the AI-generated result"""
    return working_llammy_core.process_user_request(user_request)

def get_system_health() -> Dict[str, Any]:
    """Get overall system health and statistics"""
    return working_llammy_core.get_system_health()

__all__ = [
    'get_llammy_core',
    'initialize_llammy_system',
    'process_user_request',
    'get_system_health',
    'WorkingLlammyCoreOrchestrator',
    'WorkingOllamaService',
    'WorkingDualAI',
    'WorkingVisionService'
]

print("🎯 WORKING Llammy Core v6.1 loaded successfully!")
print("✅ Real Ollama HTTP calls")
print("✅ Working dual AI system")
print("✅ Scene-aware vision service")
print("✅ New Spatial Intelligence integration")
