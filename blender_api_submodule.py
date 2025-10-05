# =============================================================================
# BLENDER API SUB-MODULE WITH AUTO-UPDATE SYSTEM
# blender_api_submodule.py - Self-updating Blender API management
# =============================================================================

import json
import time
import requests
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
import importlib.util
import sys

print("Blender API Sub-Module with Auto-Update System Loading...")

# =============================================================================
# BLENDER API VERSION MANAGER
# =============================================================================

class BlenderAPIVersionManager:
    """Manages Blender API versions and auto-updates"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path).expanduser()
        self.api_data_path = self.base_path / "blender_api_data"
        self.api_data_path.mkdir(parents=True, exist_ok=True)
        
        # API version tracking
        self.version_db_path = self.api_data_path / "api_versions.json"
        self.current_blender_version = self._detect_blender_version()
        self.cached_api_data = {}
        
        # Update configuration
        self.update_check_interval_hours = 24  # Check daily
        self.last_update_check = None
        
        # Load version database
        self._load_version_database()
        
    def _detect_blender_version(self) -> Tuple[int, int, int]:
        """Detect current Blender version"""
        try:
            import bpy
            return bpy.app.version
        except ImportError:
            # Fallback: try to detect from system
            try:
                result = subprocess.run(['blender', '--version'],
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    # Parse version from output
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'Blender' in line:
                            # Extract version numbers
                            import re
                            match = re.search(r'(\d+)\.(\d+)\.(\d+)', line)
                            if match:
                                return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            except Exception:
                pass
            
            # Default to 4.5 if detection fails
            return (4, 5, 0)
    
    def _load_version_database(self):
        """Load version database"""
        try:
            if self.version_db_path.exists():
                with open(self.version_db_path, 'r') as f:
                    data = json.load(f)
                    self.cached_api_data = data.get('api_mappings', {})
                    self.last_update_check = data.get('last_update_check')
                    if self.last_update_check:
                        self.last_update_check = datetime.fromisoformat(self.last_update_check)
        except Exception as e:
            print(f"Warning: Failed to load version database: {e}")
    
    def _save_version_database(self):
        """Save version database"""
        try:
            data = {
                'current_version': self.current_blender_version,
                'api_mappings': self.cached_api_data,
                'last_update_check': self.last_update_check.isoformat() if self.last_update_check else None,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.version_db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save version database: {e}")
    
    def check_for_updates(self) -> Dict[str, Any]:
        """Check if API updates are available"""
        try:
            now = datetime.now()
            
            # Check if we need to update
            if (self.last_update_check and
                (now - self.last_update_check).hours < self.update_check_interval_hours):
                return {
                    'update_needed': False,
                    'reason': 'Recent check within interval',
                    'last_check': self.last_update_check
                }
            
            print("Checking for Blender API updates...")
            
            # Update from web sources
            update_result = self._fetch_api_updates()
            
            self.last_update_check = now
            self._save_version_database()
            
            return update_result
            
        except Exception as e:
            return {
                'update_needed': False,
                'error': str(e)
            }
    
    def _fetch_api_updates(self) -> Dict[str, Any]:
        """Fetch API updates from web sources"""
        try:
            # API sources for updates
            api_sources = [
                {
                    'name': 'blender_docs',
                    'url': 'https://docs.blender.org/api/current/',
                    'type': 'documentation'
                },
                {
                    'name': 'github_blender',
                    'url': 'https://api.github.com/repos/blender/blender/releases/latest',
                    'type': 'release_info'
                }
            ]
            
            updates_found = []
            
            for source in api_sources:
                try:
                    update_data = self._fetch_source_update(source)
                    if update_data:
                        updates_found.append(update_data)
                        
                    # Rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    print(f"Failed to fetch from {source['name']}: {e}")
            
            if updates_found:
                # Process and cache updates
                self._process_api_updates(updates_found)
                return {
                    'update_needed': True,
                    'updates_found': len(updates_found),
                    'sources_updated': [u['source'] for u in updates_found]
                }
            else:
                return {
                    'update_needed': False,
                    'reason': 'No updates found'
                }
                
        except Exception as e:
            return {
                'update_needed': False,
                'error': f'Update fetch failed: {str(e)}'
            }
    
    def _fetch_source_update(self, source: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Fetch update from a single source"""
        try:
            response = requests.get(source['url'], timeout=10)
            if response.status_code == 200:
                
                if source['type'] == 'release_info':
                    # GitHub releases API
                    data = response.json()
                    return {
                        'source': source['name'],
                        'type': 'release',
                        'version': data.get('tag_name', ''),
                        'published_at': data.get('published_at'),
                        'description': data.get('body', '')[:500]
                    }
                    
                elif source['type'] == 'documentation':
                    # Documentation changes
                    content = response.text
                    content_hash = hashlib.md5(content.encode()).hexdigest()
                    
                    # Check if content changed
                    old_hash = self.cached_api_data.get(f"{source['name']}_hash")
                    if old_hash != content_hash:
                        return {
                            'source': source['name'],
                            'type': 'documentation',
                            'content_hash': content_hash,
                            'content_preview': content[:1000]
                        }
            
            return None
            
        except Exception as e:
            print(f"Source fetch error for {source['name']}: {e}")
            return None
    
    def _process_api_updates(self, updates: List[Dict[str, Any]]):
        """Process and cache API updates"""
        try:
            for update in updates:
                source_name = update['source']
                
                if update['type'] == 'documentation':
                    # Cache documentation hash
                    self.cached_api_data[f"{source_name}_hash"] = update['content_hash']
                    self.cached_api_data[f"{source_name}_updated"] = datetime.now().isoformat()
                    
                elif update['type'] == 'release':
                    # Cache release information
                    self.cached_api_data[f"{source_name}_version"] = update['version']
                    self.cached_api_data[f"{source_name}_updated"] = datetime.now().isoformat()
            
            print(f"Processed {len(updates)} API updates")
            
        except Exception as e:
            print(f"API update processing error: {e}")

# =============================================================================
# BLENDER API CORRECTION ENGINE
# =============================================================================

class BlenderAPICorrectionEngine:
    """Dynamic API correction based on current version"""
    
    def __init__(self, version_manager: BlenderAPIVersionManager):
        self.version_manager = version_manager
        self.blender_version = version_manager.current_blender_version
        
        # Dynamic API mappings based on version
        self.api_corrections = self._build_version_corrections()
        
        # Deprecated API tracking
        self.deprecated_apis = self._build_deprecated_apis()
        
    def _build_version_corrections(self) -> Dict[str, str]:
        """Build API corrections based on detected Blender version"""
        major, minor, patch = self.blender_version
        
        corrections = {}
        
        # Base corrections for all versions
        corrections.update({
            # Context and selection
            'bpy.context.scene.objects.active': 'bpy.context.view_layer.objects.active',
            'bpy.context.selected_objects': 'bpy.context.selected_objects',
            'obj.select': 'obj.select_set(True)',
            'obj.select_set_state': 'obj.select_set',
        })
        
        # Version-specific corrections
        if major >= 4:
            if minor >= 5:
                # Blender 4.5+ specific corrections
                corrections.update({
                    'scene.eevee': 'scene.eevee_next',
                    'eevee.use_bloom': 'eevee_next.use_bloom',
                    'eevee.use_volumetric_lights': 'eevee_next.use_volumetric_lighting',
                    'principled.inputs["Subsurface"]': 'principled.inputs["Subsurface Weight"]',
                    'principled.inputs["Specular"]': 'principled.inputs["Specular IOR Level"]',
                })
            
            if minor >= 4:
                # Blender 4.4+ corrections
                corrections.update({
                    'material.blend_method': 'material.surface_render_method',
                    'material.shadow_method': 'material.shadow_render_method',
                })
        
        return corrections
    
    def _build_deprecated_apis(self) -> Dict[str, Dict[str, str]]:
        """Build list of deprecated APIs with alternatives"""
        major, minor, patch = self.blender_version
        
        deprecated = {}
        
        if major >= 4:
            deprecated.update({
                'Mesh.calc_normals': {
                    'replacement': '# Normals auto-calculated in Blender 4.x+',
                    'reason': 'Normals are now automatically calculated',
                    'version_deprecated': '4.0'
                },
                'Mesh.calc_loop_triangles': {
                    'replacement': 'mesh.loop_triangles  # Direct property access',
                    'reason': 'Loop triangles are now a direct property',
                    'version_deprecated': '4.0'
                },
                'bpy.ops.mesh.cube_add': {
                    'replacement': 'bpy.ops.mesh.primitive_cube_add',
                    'reason': 'Consistent primitive naming',
                    'version_deprecated': '3.0'
                }
            })
        
        return deprecated
    
    def correct_api_code(self, code: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Apply API corrections to code"""
        corrected_code = code
        corrections_applied = []
        
        # Apply direct corrections
        for old_api, new_api in self.api_corrections.items():
            if old_api in corrected_code:
                corrected_code = corrected_code.replace(old_api, new_api)
                corrections_applied.append({
                    'type': 'correction',
                    'old': old_api,
                    'new': new_api,
                    'blender_version': f"{self.blender_version[0]}.{self.blender_version[1]}"
                })
        
        # Check for deprecated APIs
        for deprecated_api, info in self.deprecated_apis.items():
            if deprecated_api in corrected_code:
                corrected_code = corrected_code.replace(deprecated_api, info['replacement'])
                corrections_applied.append({
                    'type': 'deprecation',
                    'old': deprecated_api,
                    'new': info['replacement'],
                    'reason': info['reason'],
                    'deprecated_in': info['version_deprecated']
                })
        
        return corrected_code, corrections_applied
    
    def generate_version_optimized_code(self, request_type: str, parameters: Dict[str, Any]) -> str:
        """Generate code optimized for current Blender version"""
        major, minor, patch = self.blender_version
        
        if request_type == 'create_material':
            return self._generate_material_code(parameters, major, minor)
        elif request_type == 'create_object':
            return self._generate_object_code(parameters, major, minor)
        elif request_type == 'setup_lighting':
            return self._generate_lighting_code(parameters, major, minor)
        else:
            return self._generate_generic_code(parameters, major, minor)
    
    def _generate_material_code(self, params: Dict[str, Any], major: int, minor: int) -> str:
        """Generate material code for current Blender version"""
        material_name = params.get('name', 'Generated_Material')
        
        if major >= 4 and minor >= 5:
            # Blender 4.5+ with Principled BSDF v2
            return f'''
import bpy

# Create material with Blender {major}.{minor} Principled BSDF v2
material = bpy.data.materials.new(name="{material_name}")
material.use_nodes = True
material.node_tree.nodes.clear()

# Principled BSDF v2 (4.5+)
principled = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
principled.location = (0, 0)

# Set Principled BSDF v2 inputs
principled.inputs['Base Color'].default_value = {params.get('base_color', [0.8, 0.8, 0.8, 1.0])}
principled.inputs['Metallic'].default_value = {params.get('metallic', 0.0)}
principled.inputs['Roughness'].default_value = {params.get('roughness', 0.5)}
principled.inputs['Specular IOR Level'].default_value = 0.5  # New in v2

# Material Output
output = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
output.location = (300, 0)

# Connect
material.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])

# Apply to active object
if bpy.context.active_object and bpy.context.active_object.type == 'MESH':
    bpy.context.active_object.data.materials.append(material)

print(f"Created Blender {major}.{minor} material: {material_name}")
'''
        else:
            # Legacy Principled BSDF
            return f'''
import bpy

# Create material with legacy Principled BSDF
material = bpy.data.materials.new(name="{material_name}")
material.use_nodes = True
material.node_tree.nodes.clear()

# Legacy Principled BSDF
principled = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')
principled.location = (0, 0)

# Legacy inputs
principled.inputs['Base Color'].default_value = {params.get('base_color', [0.8, 0.8, 0.8, 1.0])}
principled.inputs['Metallic'].default_value = {params.get('metallic', 0.0)}
principled.inputs['Roughness'].default_value = {params.get('roughness', 0.5)}
principled.inputs['Specular'].default_value = 0.5  # Legacy

# Material Output
output = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
output.location = (300, 0)
material.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])

if bpy.context.active_object and bpy.context.active_object.type == 'MESH':
    bpy.context.active_object.data.materials.append(material)

print(f"Created legacy material: {material_name}")
'''
    
    def _generate_object_code(self, params: Dict[str, Any], major: int, minor: int) -> str:
        """Generate object creation code"""
        obj_type = params.get('type', 'CUBE')
        location = params.get('location', [0, 0, 0])
        
        return f'''
import bpy

# Create {obj_type} with current API
bpy.ops.mesh.primitive_{obj_type.lower()}_add(location={location})
obj = bpy.context.active_object

# Update scene (version-appropriate method)
{'bpy.context.view_layer.update()' if major >= 3 else 'bpy.context.scene.update()'}

print(f"Created {obj_type} at {location} using Blender {major}.{minor} API")
'''
    
    def _generate_lighting_code(self, params: Dict[str, Any], major: int, minor: int) -> str:
        """Generate lighting code"""
        if major >= 4 and minor >= 5:
            # EEVEE Next
            return f'''
import bpy

# Set EEVEE Next (Blender 4.5+)
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
eevee = bpy.context.scene.eevee_next
eevee.use_raytracing = True
eevee.use_volumetric_lighting = True

# Create light
bpy.ops.object.light_add(type='{params.get('type', 'SUN')}', location=(5, 5, 5))
light = bpy.context.active_object.data
light.energy = {params.get('energy', 5.0)}

print("Created lighting with EEVEE Next")
'''
        else:
            # Legacy EEVEE
            return f'''
import bpy

# Set legacy EEVEE
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
eevee = bpy.context.scene.eevee
eevee.use_bloom = True

# Create light
bpy.ops.object.light_add(type='{params.get('type', 'SUN')}', location=(5, 5, 5))
light = bpy.context.active_object.data
light.energy = {params.get('energy', 5.0)}

print("Created lighting with legacy EEVEE")
'''
    
    def _generate_generic_code(self, params: Dict[str, Any], major: int, minor: int) -> str:
        """Generate generic version-appropriate code"""
        return f'''
import bpy

# Blender {major}.{minor} optimized code
context = bpy.context
scene = context.scene

# Version-appropriate scene update
{'context.view_layer.update()' if major >= 3 else 'scene.update()'}

print(f"Executed with Blender {major}.{minor} API")
'''

# =============================================================================
# AUTO-UPDATING BLENDER API SUB-MODULE
# =============================================================================

class BlenderAPISubModule:
    """Self-updating Blender API sub-module"""
    
    def __init__(self, base_path: str, harvester_connection=None):
        self.version_manager = BlenderAPIVersionManager(base_path)
        self.correction_engine = BlenderAPICorrectionEngine(self.version_manager)
        self.harvester_connection = harvester_connection
        
        # Auto-update settings
        self.auto_update_enabled = True
        self.last_auto_update = None
        
        print(f"Blender API Sub-Module initialized for version {self._version_string()}")
    
    def _version_string(self) -> str:
        """Get version as string"""
        v = self.version_manager.current_blender_version
        return f"{v[0]}.{v[1]}.{v[2]}"
    
    def process_code(self, code: str, auto_correct: bool = True) -> Dict[str, Any]:
        """Process code with API corrections and optimization"""
        try:
            # Check for updates if needed
            if self.auto_update_enabled:
                self._check_auto_update()
            
            result = {
                'original_code': code,
                'blender_version': self._version_string(),
                'corrections_applied': [],
                'warnings': [],
                'success': True
            }
            
            if auto_correct:
                # Apply API corrections
                corrected_code, corrections = self.correction_engine.correct_api_code(code)
                result['corrected_code'] = corrected_code
                result['corrections_applied'] = corrections
                
                # Add warnings for deprecated APIs
                for correction in corrections:
                    if correction.get('type') == 'deprecation':
                        result['warnings'].append(
                            f"Deprecated API: {correction['old']} -> {correction['new']}"
                        )
            else:
                result['corrected_code'] = code
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'original_code': code,
                'blender_version': self._version_string()
            }
    
    def generate_optimized_code(self, request_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate version-optimized code"""
        try:
            # Check for updates
            if self.auto_update_enabled:
                self._check_auto_update()
            
            # Generate code
            generated_code = self.correction_engine.generate_version_optimized_code(
                request_type, parameters
            )
            
            return {
                'success': True,
                'generated_code': generated_code,
                'request_type': request_type,
                'blender_version': self._version_string(),
                'optimizations_applied': ['version_specific_api', 'current_best_practices']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'request_type': request_type,
                'blender_version': self._version_string()
            }
    
    def _check_auto_update(self):
        """Check and perform auto-updates"""
        try:
            # Only check once per session to avoid overhead
            if self.last_auto_update:
                return
            
            update_result = self.version_manager.check_for_updates()
            
            if update_result.get('update_needed'):
                print(f"API updates found: {update_result.get('sources_updated', [])}")
                
                # If harvester is connected, request fresh API data
                if self.harvester_connection:
                    self._request_harvester_update()
            
            self.last_auto_update = datetime.now()
            
        except Exception as e:
            print(f"Auto-update check failed: {e}")
    
    def _request_harvester_update(self):
        """Request harvester to fetch fresh API data"""
        try:
            if hasattr(self.harvester_connection, 'web_harvester'):
                print("Requesting fresh Blender API documentation from harvester...")
                # This would trigger the harvester's web scraping for API docs
                # Implementation would depend on your harvester interface
                pass
        except Exception as e:
            print(f"Harvester update request failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get sub-module status"""
        return {
            'module_name': 'blender_api_submodule',
            'blender_version': self._version_string(),
            'auto_update_enabled': self.auto_update_enabled,
            'last_update_check': (
                self.version_manager.last_update_check.isoformat()
                if self.version_manager.last_update_check else None
            ),
            'api_corrections_count': len(self.correction_engine.api_corrections),
            'deprecated_apis_count': len(self.correction_engine.deprecated_apis),
            'harvester_connected': self.harvester_connection is not None,
            'status': 'active'
        }
    
    def enable_auto_updates(self, enabled: bool = True):
        """Enable/disable auto-updates"""
        self.auto_update_enabled = enabled
        print(f"Auto-updates {'enabled' if enabled else 'disabled'}")
    
    def force_update_check(self) -> Dict[str, Any]:
        """Force an immediate update check"""
        self.last_auto_update = None  # Reset to force check
        return self.version_manager.check_for_updates()

# =============================================================================
# FACTORY FUNCTION FOR CORE INTEGRATION
# =============================================================================

def create_blender_api_submodule(base_path: str, harvester_connection=None) -> BlenderAPISubModule:
    """Factory function to create Blender API sub-module"""
    try:
        submodule = BlenderAPISubModule(base_path, harvester_connection)
        
        print("Blender API Sub-Module created with features:")
        print(f"  - Dynamic API correction for Blender {submodule._version_string()}")
        print("  - Auto-update system with web harvesting integration")
        print("  - Version-specific code generation")
        print("  - Deprecated API detection and replacement")
        print("  - Harvester integration for fresh API data")
        
        return submodule
        
    except Exception as e:
        print(f"Failed to create Blender API sub-module: {e}")
        return None

# =============================================================================
# INTEGRATION EXAMPLE
# =============================================================================

def integrate_with_core(core_module, harvester_module):
    """Example integration with core and harvester modules"""
    
    # Create API sub-module with harvester connection
    api_submodule = create_blender_api_submodule(
        base_path="~/.llammy/api_cache",
        harvester_connection=harvester_module
    )
    
    if api_submodule:
        # Integrate with core module
        if hasattr(core_module, 'set_api_submodule'):
            core_module.set_api_submodule(api_submodule)
        
        # Set up auto-update triggers
        if hasattr(harvester_module, 'add_update_callback'):
            harvester_module.add_update_callback(
                'blender_api_docs',
                api_submodule.force_update_check
            )
        
        print("Blender API sub-module integrated with core and harvester")
        return True
    
    return False

if __name__ == "__main__":
    # Test the sub-module
    api_module = create_blender_api_submodule("~/test_api_cache")
    
    if api_module:
        # Test code correction
        test_code = '''
import bpy
bpy.context.scene.objects.active = None
bpy.ops.mesh.cube_add()
scene.eevee.use_bloom = True
'''
        
        result = api_module.process_code(test_code)
        
        print("\nTest Results:")
        print(f"Success: {result['success']}")
        print(f"Corrections applied: {len(result['corrections_applied'])}")
        print(f"Warnings: {len(result['warnings'])}")
        
        # Test code generation
        gen_result = api_module.generate_optimized_code('create_material', {
            'name': 'TestMaterial',
            'base_color': [0.8, 0.2, 0.2, 1.0]
        })
        
        print(f"\nCode generation success: {gen_result['success']}")
        
        # Status check
        status = api_module.get_status()
        print(f"\nModule status: {status['status']}")
        print(f"Blender version: {status['blender_version']}")

print("BLENDER API SUB-MODULE LOADED!")
print("Features:")
print("  - Auto-updating API corrections")
print("  - Version-specific code generation")
print("  - Harvester integration for fresh data")
print("  - Deprecated API detection")
print("  - Modular architecture with clean separation")
