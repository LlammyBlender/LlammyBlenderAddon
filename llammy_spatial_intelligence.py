# =============================================================================
# LLAMMY SPATIAL INTELLIGENCE MODULE v1.0
# llammy_spatial_intelligence.py - Scene Graph Engine with MCP Integration
# Integrates with your existing MCP architecture for spatial AI reasoning
# =============================================================================

import bpy
import bmesh
import mathutils
import numpy as np
import json
import time
import sqlite3
import os
import requests
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading
import queue

print("🌌 Llammy Spatial Intelligence Module v1.0 - Scene Graph Engine")

# =============================================================================
# SPATIAL DATA STRUCTURES - INSPIRED BY THE MEDIUM ARTICLE
# =============================================================================

@dataclass
class SpatialNode:
    """Spatial node representing an object/entity in 3D scene graph"""
    id: str
    name: str
    object_type: str  # MESH, LIGHT, CAMERA, EMPTY, etc.
    location: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]  # Quaternion
    scale: Tuple[float, float, float]
    bounding_box: Dict[str, Tuple[float, float, float]]  # min, max, center, dimensions
    semantic_tags: List[str]  # ['character', 'prop', 'environment', etc.]
    properties: Dict[str, Any]  # Material info, physics, etc.
    timestamp: float

@dataclass 
class SpatialRelationship:
    """Spatial relationship between nodes"""
    id: str
    source_node: str
    target_node: str
    relationship_type: str  # NEAR, FAR, ABOVE, BELOW, INSIDE, PARENT_OF, etc.
    distance: float
    confidence: float  # 0.0-1.0
    semantic_meaning: str  # Human-readable description
    metadata: Dict[str, Any]
    timestamp: float

class SpatialSceneGraph:
    """3D Scene graph for spatial AI reasoning"""
    
    def __init__(self):
        self.nodes: Dict[str, SpatialNode] = {}
        self.relationships: Dict[str, SpatialRelationship] = {}
        self.spatial_index = {}  # For fast spatial queries
        self.semantic_groups = defaultdict(list)
        self.version = 1.0
        self.last_update = time.time()
        
        print("🔗 Spatial Scene Graph initialized")
    
    def add_node(self, node: SpatialNode):
        """Add spatial node to scene graph"""
        self.nodes[node.id] = node
        
        # Update semantic groups
        for tag in node.semantic_tags:
            if node.id not in self.semantic_groups[tag]:
                self.semantic_groups[tag].append(node.id)
        
        # Update spatial index for fast queries
        self._update_spatial_index(node)
        self.last_update = time.time()
    
    def add_relationship(self, relationship: SpatialRelationship):
        """Add spatial relationship"""
        self.relationships[relationship.id] = relationship
        self.last_update = time.time()
    
    def query_spatial_relationships(self, node_id: str, relationship_types: List[str] = None) -> List[SpatialRelationship]:
        """Query spatial relationships for a node"""
        results = []
        for rel in self.relationships.values():
            if rel.source_node == node_id or rel.target_node == node_id:
                if not relationship_types or rel.relationship_type in relationship_types:
                    results.append(rel)
        return results
    
    def find_nodes_by_semantic_tag(self, tag: str) -> List[SpatialNode]:
        """Find nodes by semantic tag"""
        node_ids = self.semantic_groups.get(tag, [])
        return [self.nodes[node_id] for node_id in node_ids if node_id in self.nodes]
    
    def get_scene_context_summary(self) -> Dict[str, Any]:
        """Get comprehensive scene context for AI reasoning"""
        return {
            'total_nodes': len(self.nodes),
            'total_relationships': len(self.relationships),
            'semantic_groups': {tag: len(nodes) for tag, nodes in self.semantic_groups.items()},
            'scene_bounds': self._calculate_scene_bounds(),
            'complexity_score': self._calculate_complexity_score(),
            'last_update': self.last_update
        }
    
    def _update_spatial_index(self, node: SpatialNode):
        """Update spatial index for fast spatial queries"""
        # Simple grid-based spatial index
        grid_size = 5.0  # 5 Blender units per grid cell
        grid_x = int(node.location[0] // grid_size)
        grid_y = int(node.location[1] // grid_size)
        grid_z = int(node.location[2] // grid_size)
        
        grid_key = f"{grid_x},{grid_y},{grid_z}"
        if grid_key not in self.spatial_index:
            self.spatial_index[grid_key] = []
        
        if node.id not in self.spatial_index[grid_key]:
            self.spatial_index[grid_key].append(node.id)
    
    def _calculate_scene_bounds(self) -> Dict[str, Tuple[float, float, float]]:
        """Calculate overall scene bounding box"""
        if not self.nodes:
            return {'min': (0, 0, 0), 'max': (0, 0, 0), 'center': (0, 0, 0)}
        
        locations = [node.location for node in self.nodes.values()]
        
        min_x = min(loc[0] for loc in locations)
        min_y = min(loc[1] for loc in locations)
        min_z = min(loc[2] for loc in locations)
        
        max_x = max(loc[0] for loc in locations)
        max_y = max(loc[1] for loc in locations)
        max_z = max(loc[2] for loc in locations)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2
        
        return {
            'min': (min_x, min_y, min_z),
            'max': (max_x, max_y, max_z),
            'center': (center_x, center_y, center_z),
            'dimensions': (max_x - min_x, max_y - min_y, max_z - min_z)
        }
    
    def _calculate_complexity_score(self) -> float:
        """Calculate scene complexity score for AI reasoning"""
        base_score = len(self.nodes)
        relationship_bonus = len(self.relationships) * 0.5
        semantic_diversity = len(self.semantic_groups) * 2
        
        return base_score + relationship_bonus + semantic_diversity
    
    def to_networkx_graph(self):
        """Export to NetworkX format (if available) for advanced analysis"""
        try:
            import networkx as nx
            
            G = nx.DiGraph()
            
            # Add nodes
            for node_id, node in self.nodes.items():
                G.add_node(node_id, **asdict(node))
            
            # Add relationships as edges
            for rel_id, rel in self.relationships.items():
                G.add_edge(
                    rel.source_node,
                    rel.target_node,
                    relationship_type=rel.relationship_type,
                    distance=rel.distance,
                    confidence=rel.confidence,
                    semantic_meaning=rel.semantic_meaning
                )
            
            return G
        except ImportError:
            print("⚠️ NetworkX not available for advanced graph analysis")
            return None

# =============================================================================
# BLENDER SCENE ANALYZER - INTEGRATES WITH YOUR VISION SERVICE
# =============================================================================

class BlenderSceneAnalyzer:
    """Analyzes Blender scenes to build spatial intelligence"""
    
    def __init__(self):
        self.supported_types = {'MESH', 'CAMERA', 'LIGHT', 'EMPTY', 'ARMATURE', 'CURVE'}
        self.semantic_classifiers = {
            'character': ['human', 'person', 'character', 'figure', 'body'],
            'vehicle': ['car', 'truck', 'bike', 'vehicle', 'transport'],
            'building': ['house', 'building', 'structure', 'wall', 'floor'],
            'nature': ['tree', 'plant', 'rock', 'terrain', 'landscape'],
            'furniture': ['chair', 'table', 'desk', 'bed', 'shelf'],
            'lighting': ['light', 'lamp', 'sun', 'spot', 'area']
        }
        
        print("🔍 Blender Scene Analyzer ready")
    
    def analyze_current_scene(self) -> SpatialSceneGraph:
        """Analyze current Blender scene and build spatial graph"""
        scene_graph = SpatialSceneGraph()
        
        try:
            scene = bpy.context.scene
            
            # Process all objects
            for obj in scene.objects:
                if obj.type in self.supported_types:
                    spatial_node = self._create_spatial_node(obj)
                    scene_graph.add_node(spatial_node)
            
            # Calculate spatial relationships
            self._calculate_spatial_relationships(scene_graph)
            
            print(f"🌐 Scene graph built: {len(scene_graph.nodes)} nodes, {len(scene_graph.relationships)} relationships")
            
        except Exception as e:
            print(f"❌ Scene analysis failed: {e}")
        
        return scene_graph
    
    def _create_spatial_node(self, obj) -> SpatialNode:
        """Create spatial node from Blender object"""
        
        # Get object transforms
        location = tuple(obj.location)
        rotation = tuple(obj.rotation_quaternion)
        scale = tuple(obj.scale)
        
        # Calculate bounding box
        bbox = self._calculate_bounding_box(obj)
        
        # Semantic classification
        semantic_tags = self._classify_semantically(obj)
        
        # Object properties
        properties = {
            'blender_type': obj.type,
            'visible': obj.visible_get(),
            'parent': obj.parent.name if obj.parent else None,
            'material_count': len(obj.material_slots) if hasattr(obj, 'material_slots') else 0
        }
        
        # Add material information
        if hasattr(obj, 'material_slots') and len(obj.material_slots) > 0:
            materials = []
            for slot in obj.material_slots:
                if slot.material:
                    materials.append({
                        'name': slot.material.name,
                        'type': 'principled' if slot.material.use_nodes else 'legacy'
                    })
            properties['materials'] = materials
        
        return SpatialNode(
            id=obj.name,
            name=obj.name,
            object_type=obj.type,
            location=location,
            rotation=rotation,
            scale=scale,
            bounding_box=bbox,
            semantic_tags=semantic_tags,
            properties=properties,
            timestamp=time.time()
        )
    
    def _calculate_bounding_box(self, obj) -> Dict[str, Tuple[float, float, float]]:
        """Calculate object bounding box"""
        if obj.type == 'MESH':
            # Get mesh bounding box in world coordinates
            bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
            
            min_x = min(corner.x for corner in bbox_corners)
            min_y = min(corner.y for corner in bbox_corners)
            min_z = min(corner.z for corner in bbox_corners)
            
            max_x = max(corner.x for corner in bbox_corners)
            max_y = max(corner.y for corner in bbox_corners)
            max_z = max(corner.z for corner in bbox_corners)
            
            center = ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
            dimensions = (max_x - min_x, max_y - min_y, max_z - min_z)
            
            return {
                'min': (min_x, min_y, min_z),
                'max': (max_x, max_y, max_z),
                'center': center,
                'dimensions': dimensions
            }
        else:
            # For non-mesh objects, use object location as center
            loc = tuple(obj.location)
            return {
                'min': loc,
                'max': loc,
                'center': loc,
                'dimensions': (0, 0, 0)
            }
    
    def _classify_semantically(self, obj) -> List[str]:
        """Classify object semantically based on name and properties"""
        tags = []
        name_lower = obj.name.lower()
        
        # Object type tag
        tags.append(obj.type.lower())
        
        # Semantic classification based on name
        for category, keywords in self.semantic_classifiers.items():
            if any(keyword in name_lower for keyword in keywords):
                tags.append(category)
        
        # Special cases based on object properties
        if obj.type == 'LIGHT':
            tags.append('lighting')
        elif obj.type == 'CAMERA':
            tags.append('camera')
        elif obj.parent:
            tags.append('child_object')
        
        return tags if tags else ['unknown']
    
    def _calculate_spatial_relationships(self, scene_graph: SpatialSceneGraph):
        """Calculate spatial relationships between objects"""
        nodes = list(scene_graph.nodes.values())
        
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i+1:]:
                relationship = self._analyze_spatial_relationship(node_a, node_b)
                if relationship:
                    scene_graph.add_relationship(relationship)
    
    def _analyze_spatial_relationship(self, node_a: SpatialNode, node_b: SpatialNode) -> Optional[SpatialRelationship]:
        """Analyze spatial relationship between two nodes"""
        
        # Calculate distance
        distance = self._calculate_distance(node_a.location, node_b.location)
        
        # Skip if too far apart
        if distance > 50.0:  # 50 Blender units
            return None
        
        # Determine relationship type
        relationship_type, confidence, semantic_meaning = self._determine_relationship_type(
            node_a, node_b, distance
        )
        
        if relationship_type:
            return SpatialRelationship(
                id=f"{node_a.id}_to_{node_b.id}",
                source_node=node_a.id,
                target_node=node_b.id,
                relationship_type=relationship_type,
                distance=distance,
                confidence=confidence,
                semantic_meaning=semantic_meaning,
                metadata={
                    'source_type': node_a.object_type,
                    'target_type': node_b.object_type
                },
                timestamp=time.time()
            )
        
        return None
    
    def _calculate_distance(self, loc_a: Tuple[float, float, float], loc_b: Tuple[float, float, float]) -> float:
        """Calculate 3D distance between two locations"""
        return ((loc_a[0] - loc_b[0])**2 + (loc_a[1] - loc_b[1])**2 + (loc_a[2] - loc_b[2])**2)**0.5
    
    def _determine_relationship_type(self, node_a: SpatialNode, node_b: SpatialNode, distance: float) -> Tuple[Optional[str], float, str]:
        """Determine spatial relationship type between nodes"""
        
        loc_a = node_a.location
        loc_b = node_b.location
        
        # Vertical relationships
        height_diff = abs(loc_a[2] - loc_b[2])
        if height_diff > 0.5:  # Significant height difference
            if loc_a[2] > loc_b[2]:
                return "ABOVE", 0.9, f"{node_a.name} is above {node_b.name}"
            else:
                return "BELOW", 0.9, f"{node_a.name} is below {node_b.name}"
        
        # Horizontal proximity relationships
        if distance < 2.0:
            return "NEAR", 0.8, f"{node_a.name} is near {node_b.name}"
        elif distance < 5.0:
            return "NEARBY", 0.6, f"{node_a.name} is nearby {node_b.name}"
        elif distance < 15.0:
            return "DISTANT", 0.4, f"{node_a.name} is distant from {node_b.name}"
        
        return None, 0.0, ""

# =============================================================================
# SPATIAL QUERY ENGINE - INTEGRATES WITH YOUR AI MODULES
# =============================================================================

class SpatialQueryEngine:
    """Query engine for spatial reasoning - integrates with your MCP system"""
    
    def __init__(self, scene_graph: SpatialSceneGraph):
        self.scene_graph = scene_graph
        self.query_cache = {}
        self.cache_timeout = 30  # seconds
        
        print("🔍 Spatial Query Engine initialized")
    
    def query_path_clear(self, start_object: str, end_object: str, buffer_distance: float = 1.0) -> Dict[str, Any]:
        """Query if path between objects is clear"""
        cache_key = f"path_{start_object}_{end_object}_{buffer_distance}"
        
        if self._is_cached(cache_key):
            return self.query_cache[cache_key]['result']
        
        if start_object not in self.scene_graph.nodes or end_object not in self.scene_graph.nodes:
            return {'clear': False, 'error': 'Object not found'}
        
        start_node = self.scene_graph.nodes[start_object]
        end_node = self.scene_graph.nodes[end_object]
        
        # Check for objects in the path
        obstructing_objects = []
        
        for node_id, node in self.scene_graph.nodes.items():
            if node_id in [start_object, end_object]:
                continue
            
            if self._is_in_path(start_node.location, end_node.location, node.location, buffer_distance):
                obstructing_objects.append({
                    'name': node.name,
                    'type': node.object_type,
                    'location': node.location
                })
        
        result = {
            'clear': len(obstructing_objects) == 0,
            'obstructing_objects': obstructing_objects,
            'path_distance': self._calculate_distance(start_node.location, end_node.location),
            'query_timestamp': time.time()
        }
        
        self._cache_result(cache_key, result)
        return result
    
    def query_objects_near(self, target_object: str, radius: float = 5.0, object_types: List[str] = None) -> List[Dict[str, Any]]:
        """Query objects near a target object"""
        if target_object not in self.scene_graph.nodes:
            return []
        
        target_node = self.scene_graph.nodes[target_object]
        nearby_objects = []
        
        for node_id, node in self.scene_graph.nodes.items():
            if node_id == target_object:
                continue
            
            distance = self._calculate_distance(target_node.location, node.location)
            
            if distance <= radius:
                if not object_types or node.object_type in object_types:
                    nearby_objects.append({
                        'name': node.name,
                        'type': node.object_type,
                        'distance': distance,
                        'location': node.location,
                        'semantic_tags': node.semantic_tags
                    })
        
        # Sort by distance
        nearby_objects.sort(key=lambda x: x['distance'])
        return nearby_objects
    
    def query_spatial_constraints(self, proposed_location: Tuple[float, float, float], object_type: str = 'MESH') -> Dict[str, Any]:
        """Query spatial constraints for placing new object"""
        constraints = []
        warnings = []
        suggestions = []
        
        # Check for collisions
        collision_objects = []
        for node in self.scene_graph.nodes.values():
            distance = self._calculate_distance(proposed_location, node.location)
            
            # Simple collision detection based on bounding box
            if node.object_type == 'MESH':
                dimensions = node.bounding_box.get('dimensions', (1, 1, 1))
                min_distance = max(dimensions) / 2 + 0.5  # Buffer
                
                if distance < min_distance:
                    collision_objects.append({
                        'name': node.name,
                        'distance': distance,
                        'min_safe_distance': min_distance
                    })
        
        if collision_objects:
            constraints.append('COLLISION_RISK')
            warnings.append(f"Potential collision with {len(collision_objects)} objects")
        
        # Check if location is reasonable
        scene_bounds = self.scene_graph._calculate_scene_bounds()
        if self._is_location_outside_scene(proposed_location, scene_bounds):
            warnings.append("Location is outside main scene area")
            
            # Suggest better location
            center = scene_bounds['center']
            suggestions.append({
                'type': 'ALTERNATIVE_LOCATION',
                'location': (center[0] + 2.0, center[1], center[2]),
                'reason': 'Closer to scene center'
            })
        
        return {
            'safe_to_place': len(constraints) == 0,
            'constraints': constraints,
            'warnings': warnings,
            'suggestions': suggestions,
            'collision_objects': collision_objects
        }
    
    def query_semantic_context(self, query: str) -> Dict[str, Any]:
        """Query scene for semantic context relevant to AI request"""
        query_lower = query.lower()
        relevant_objects = []
        scene_context = self.scene_graph.get_scene_context_summary()
        
        # Find semantically relevant objects
        for node in self.scene_graph.nodes.values():
            relevance_score = 0.0
            
            # Name matching
            if any(word in node.name.lower() for word in query_lower.split()):
                relevance_score += 0.5
            
            # Semantic tag matching
            for tag in node.semantic_tags:
                if tag in query_lower:
                    relevance_score += 0.3
            
            # Object type matching
            if node.object_type.lower() in query_lower:
                relevance_score += 0.2
            
            if relevance_score > 0.0:
                relevant_objects.append({
                    'name': node.name,
                    'type': node.object_type,
                    'semantic_tags': node.semantic_tags,
                    'location': node.location,
                    'relevance_score': relevance_score
                })
        
        # Sort by relevance
        relevant_objects.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return {
            'relevant_objects': relevant_objects[:10],  # Top 10
            'scene_context': scene_context,
            'query_processed': query,
            'total_matches': len(relevant_objects)
        }
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if query result is cached and still valid"""
        if cache_key in self.query_cache:
            cache_entry = self.query_cache[cache_key]
            return (time.time() - cache_entry['timestamp']) < self.cache_timeout
        return False
    
    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache query result"""
        self.query_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def _is_in_path(self, start: Tuple[float, float, float], end: Tuple[float, float, float], 
                   point: Tuple[float, float, float], buffer: float) -> bool:
        """Check if point is in path between start and end with buffer"""
        # Simple line-point distance calculation
        start_vec = mathutils.Vector(start)
        end_vec = mathutils.Vector(end)
        point_vec = mathutils.Vector(point)
        
        line_vec = end_vec - start_vec
        point_to_start = point_vec - start_vec
        
        if line_vec.length == 0:
            return False
        
        # Project point onto line
        t = point_to_start.dot(line_vec) / line_vec.length_squared
        
        # Clamp t to line segment
        t = max(0, min(1, t))
        
        # Find closest point on line
        closest = start_vec + t * line_vec
        distance = (point_vec - closest).length
        
        return distance <= buffer
    
    def _calculate_distance(self, loc_a: Tuple[float, float, float], loc_b: Tuple[float, float, float]) -> float:
        """Calculate 3D distance"""
        return ((loc_a[0] - loc_b[0])**2 + (loc_a[1] - loc_b[1])**2 + (loc_a[2] - loc_b[2])**2)**0.5
    
    def _is_location_outside_scene(self, location: Tuple[float, float, float], scene_bounds: Dict[str, Any]) -> bool:
        """Check if location is outside reasonable scene bounds"""
        min_bounds = scene_bounds['min']
        max_bounds = scene_bounds['max']
        
        buffer = 10.0  # 10 unit buffer
        
        return (location[0] < min_bounds[0] - buffer or location[0] > max_bounds[0] + buffer or
                location[1] < min_bounds[1] - buffer or location[1] > max_bounds[1] + buffer or
                location[2] < min_bounds[2] - buffer or location[2] > max_bounds[2] + buffer)

# =============================================================================
# MCP COMMUNICATION INTERFACE - INTEGRATES WITH YOUR ARCHITECTURE
# =============================================================================

class SpatialIntelligenceMCP:
    """MCP communication interface for spatial intelligence module"""
    
    def __init__(self, ollama_service=None, embedded_model=None):
        self.version = "1.0.0"
        self.module_id = "spatial_intelligence"
        self.scene_analyzer = BlenderSceneAnalyzer()
        self.current_scene_graph = None
        self.query_engine = None
        self.ollama_service = ollama_service
        self.embedded_model = embedded_model
        
        # Statistics
        self.stats = {
            'scenes_analyzed': 0,
            'queries_processed': 0,
            'spatial_enhancements_provided': 0,
            'ai_integrations': 0,
            'uptime_start': time.time()
        }
        
        # Initialize with current scene
        self.refresh_scene_graph()
        
        print(f"🌌 Spatial Intelligence MCP Module v{self.version} initialized")
    
    def refresh_scene_graph(self):
        """Refresh scene graph from current Blender scene"""
        try:
            self.current_scene_graph = self.scene_analyzer.analyze_current_scene()
            self.query_engine = SpatialQueryEngine(self.current_scene_graph)
            self.stats['scenes_analyzed'] += 1
            
            print(f"🔄 Scene graph refreshed: {len(self.current_scene_graph.nodes)} objects")
            return True
        except Exception as e:
            print(f"❌ Scene graph refresh failed: {e}")
            return False
    
    def process_mcp_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process MCP message from other modules"""
        message_type = message.get('type')
        data = message.get('data', {})
        
        if message_type == 'SPATIAL_QUERY':
            return self._handle_spatial_query(data)
        elif message_type == 'SCENE_CONTEXT_REQUEST':
            return self._handle_scene_context_request(data)
        elif message_type == 'PLACEMENT_VALIDATION':
            return self._handle_placement_validation(data)
        elif message_type == 'PATH_CHECK':
            return self._handle_path_check(data)
        elif message_type == 'AI_ENHANCEMENT_REQUEST':
            return self._handle_ai_enhancement_request(data)
        else:
            return {
                'success': False,
                'error': f'Unknown message type: {message_type}',
                'module': self.module_id
            }
    
    def _handle_spatial_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle spatial query request"""
        if not self.query_engine:
            return {'success': False, 'error': 'Scene graph not initialized'}
        
        query_type = data.get('query_type')
        self.stats['queries_processed'] += 1
        
        try:
            if query_type == 'objects_near':
                result = self.query_engine.query_objects_near(
                    data.get('target_object', ''),
                    data.get('radius', 5.0),
                    data.get('object_types', None)
                )
            elif query_type == 'path_clear':
                result = self.query_engine.query_path_clear(
                    data.get('start_object', ''),
                    data.get('end_object', ''),
                    data.get('buffer_distance', 1.0)
                )
            elif query_type == 'semantic_context':
                result = self.query_engine.query_semantic_context(
                    data.get('query', '')
                )
            else:
                return {'success': False, 'error': f'Unknown query type: {query_type}'}
            
            return {
                'success': True,
                'result': result,
                'query_type': query_type,
                'module': self.module_id
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'query_type': query_type,
                'module': self.module_id
            }
    
    def _handle_scene_context_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle scene context request for AI modules"""
        if not self.current_scene_graph:
            return {'success': False, 'error': 'Scene graph not available'}
        
        try:
            context_summary = self.current_scene_graph.get_scene_context_summary()
            
            # Add spatial intelligence enhancements
            enhanced_context = {
                'scene_summary': context_summary,
                'spatial_recommendations': self._generate_spatial_recommendations(data.get('user_request', '')),
                'object_relationships': self._get_relationship_summary(),
                'placement_guidance': self._get_placement_guidance(),
                'scene_graph_version': self.current_scene_graph.version
            }
            
            self.stats['spatial_enhancements_provided'] += 1
            
            return {
                'success': True,
                'enhanced_context': enhanced_context,
                'module': self.module_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'module': self.module_id
            }
    
    def _handle_placement_validation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle placement validation request"""
        if not self.query_engine:
            return {'success': False, 'error': 'Query engine not available'}
        
        try:
            proposed_location = data.get('location', [0, 0, 0])
            object_type = data.get('object_type', 'MESH')
            
            constraints = self.query_engine.query_spatial_constraints(
                tuple(proposed_location), object_type
            )
            
            return {
                'success': True,
                'constraints': constraints,
                'module': self.module_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'module': self.module_id
            }
    
    def _handle_path_check(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle path checking request"""
        if not self.query_engine:
            return {'success': False, 'error': 'Query engine not available'}
        
        try:
            start_obj = data.get('start_object', '')
            end_obj = data.get('end_object', '')
            buffer = data.get('buffer_distance', 1.0)
            
            path_result = self.query_engine.query_path_clear(start_obj, end_obj, buffer)
            
            return {
                'success': True,
                'path_result': path_result,
                'module': self.module_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'module': self.module_id
            }
    
    def _handle_ai_enhancement_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle AI enhancement request from other modules"""
        if not self.query_engine:
            return {'success': False, 'error': 'Query engine not available'}
        
        try:
            user_request = data.get('user_request', '')
            
            # Get semantic context for AI reasoning
            semantic_context = self.query_engine.query_semantic_context(user_request)
            
            # Generate spatial enhancements using embedded model if available
            ai_enhancements = {}
            if self.embedded_model:
                try:
                    scene_data = {
                        'object_count': len(self.current_scene_graph.nodes),
                        'selected_objects': [],
                        'complexity': semantic_context['scene_context']['complexity_score']
                    }
                    
                    vision_assist = self.embedded_model.assist_vision_context(scene_data, user_request)
                    ai_enhancements['vision_assist'] = vision_assist
                    
                except Exception as e:
                    print(f"Embedded model enhancement failed: {e}")
            
            self.stats['ai_integrations'] += 1
            
            return {
                'success': True,
                'semantic_context': semantic_context,
                'ai_enhancements': ai_enhancements,
                'spatial_intelligence_applied': True,
                'module': self.module_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'module': self.module_id
            }
    
    def _generate_spatial_recommendations(self, user_request: str) -> List[Dict[str, Any]]:
        """Generate spatial recommendations based on user request and scene"""
        recommendations = []
        
        if not self.current_scene_graph:
            return recommendations
        
        # Analyze request for spatial keywords
        request_lower = user_request.lower()
        
        # Placement recommendations
        if any(word in request_lower for word in ['create', 'add', 'place', 'put']):
            scene_bounds = self.current_scene_graph._calculate_scene_bounds()
            center = scene_bounds['center']
            
            recommendations.append({
                'type': 'placement',
                'recommendation': f"Consider placing new objects near scene center at {center}",
                'reasoning': 'Maintains visual balance and avoids edge placement'
            })
        
        # Animation recommendations
        if any(word in request_lower for word in ['animate', 'move', 'rotate', 'keyframe']):
            moving_objects = []
            for node in self.current_scene_graph.nodes.values():
                if 'character' in node.semantic_tags or 'vehicle' in node.semantic_tags:
                    moving_objects.append(node.name)
            
            if moving_objects:
                recommendations.append({
                    'type': 'animation',
                    'recommendation': f"Consider animating these objects: {', '.join(moving_objects[:3])}",
                    'reasoning': 'These objects are semantically suitable for animation'
                })
        
        # Lighting recommendations
        lights = self.current_scene_graph.find_nodes_by_semantic_tag('lighting')
        if len(lights) < 2 and 'render' in request_lower:
            recommendations.append({
                'type': 'lighting',
                'recommendation': 'Consider adding more light sources for better rendering',
                'reasoning': f'Scene currently has {len(lights)} light sources'
            })
        
        return recommendations
    
    def _get_relationship_summary(self) -> Dict[str, Any]:
        """Get summary of spatial relationships in scene"""
        if not self.current_scene_graph:
            return {}
        
        relationship_types = {}
        for rel in self.current_scene_graph.relationships.values():
            rel_type = rel.relationship_type
            if rel_type not in relationship_types:
                relationship_types[rel_type] = 0
            relationship_types[rel_type] += 1
        
        return {
            'total_relationships': len(self.current_scene_graph.relationships),
            'relationship_types': relationship_types,
            'most_connected_objects': self._find_most_connected_objects()
        }
    
    def _find_most_connected_objects(self) -> List[Dict[str, Any]]:
        """Find objects with most spatial relationships"""
        if not self.current_scene_graph:
            return []
        
        object_connections = {}
        
        for rel in self.current_scene_graph.relationships.values():
            # Count connections for source object
            if rel.source_node not in object_connections:
                object_connections[rel.source_node] = 0
            object_connections[rel.source_node] += 1
            
            # Count connections for target object
            if rel.target_node not in object_connections:
                object_connections[rel.target_node] = 0
            object_connections[rel.target_node] += 1
        
        # Sort by connection count
        sorted_objects = sorted(object_connections.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'name': obj_name, 'connection_count': count}
            for obj_name, count in sorted_objects[:5]  # Top 5
        ]
    
    def _get_placement_guidance(self) -> Dict[str, Any]:
        """Get placement guidance for new objects"""
        if not self.current_scene_graph:
            return {'guidance': 'Center objects at origin (0, 0, 0)'}
        
        scene_bounds = self.current_scene_graph._calculate_scene_bounds()
        object_count = len(self.current_scene_graph.nodes)
        
        if object_count == 0:
            return {
                'guidance': 'Scene is empty - place objects at origin',
                'suggested_location': [0, 0, 0]
            }
        else:
            # Suggest placement away from existing objects
            center = scene_bounds['center']
            dimensions = scene_bounds['dimensions']
            max_dimension = max(dimensions)
            
            offset_distance = max(2.0, max_dimension * 0.3)
            
            return {
                'guidance': f'Place objects {offset_distance:.1f} units away from existing objects',
                'suggested_location': [center[0] + offset_distance, center[1], center[2]],
                'scene_center': center,
                'scene_dimensions': dimensions
            }
    
    def get_module_status(self) -> Dict[str, Any]:
        """Get comprehensive module status"""
        uptime = time.time() - self.stats['uptime_start']
        
        status = {
            'module_id': self.module_id,
            'version': self.version,
            'uptime_seconds': uptime,
            'statistics': self.stats.copy(),
            'scene_graph_active': self.current_scene_graph is not None,
            'query_engine_active': self.query_engine is not None,
            'services_connected': {
                'ollama_service': self.ollama_service is not None,
                'embedded_model': self.embedded_model is not None
            }
        }
        
        if self.current_scene_graph:
            status['scene_graph_status'] = self.current_scene_graph.get_scene_context_summary()
        
        return status
    
    def integrate_with_existing_modules(self, llammy_core, file_manager=None, model_manager=None):
        """Integration method for your existing module architecture"""
        print(f"🔗 Integrating Spatial Intelligence with existing Llammy modules...")
        
        # Connect to Ollama service if available
        if hasattr(llammy_core, 'ollama_service'):
            self.ollama_service = llammy_core.ollama_service
            print("✅ Connected to Ollama service")
        
        # Connect to embedded model if available
        if file_manager and hasattr(file_manager, 'embedded_model'):
            self.embedded_model = file_manager.embedded_model
            print("✅ Connected to embedded model")
        
        # Register message handlers with core MCP system
        if hasattr(llammy_core, 'register_module_handler'):
            llammy_core.register_module_handler(self.module_id, self.process_mcp_message)
            print("✅ Registered with MCP message system")
        
        print(f"🌌 Spatial Intelligence Module fully integrated")
        return True

# =============================================================================
# GLOBAL INTERFACE FUNCTIONS - COMPATIBLE WITH YOUR ARCHITECTURE
# =============================================================================

# Global spatial intelligence instance
_spatial_intelligence_module = None

def get_spatial_intelligence_module() -> SpatialIntelligenceMCP:
    """Get global spatial intelligence module"""
    global _spatial_intelligence_module
    if _spatial_intelligence_module is None:
        _spatial_intelligence_module = SpatialIntelligenceMCP()
    return _spatial_intelligence_module

def initialize_spatial_intelligence(llammy_core=None, file_manager=None, model_manager=None) -> Dict[str, Any]:
    """Initialize spatial intelligence module with your existing services"""
    try:
        module = get_spatial_intelligence_module()
        
        # Integrate with existing modules
        if llammy_core:
            module.integrate_with_existing_modules(llammy_core, file_manager, model_manager)
        
        # Initial scene analysis
        success = module.refresh_scene_graph()
        
        return {
            'success': success,
            'module_id': module.module_id,
            'version': module.version,
            'scene_objects': len(module.current_scene_graph.nodes) if module.current_scene_graph else 0,
            'spatial_relationships': len(module.current_scene_graph.relationships) if module.current_scene_graph else 0
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'module_id': 'spatial_intelligence'
        }

def query_spatial_intelligence(query_type: str, **params) -> Dict[str, Any]:
    """Query spatial intelligence module"""
    module = get_spatial_intelligence_module()
    
    message = {
        'type': query_type,
        'data': params
    }
    
    return module.process_mcp_message(message)

def enhance_ai_request_with_spatial_context(user_request: str, scene_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Enhance AI request with spatial intelligence - main integration point"""
    module = get_spatial_intelligence_module()
    
    # Refresh scene graph if needed
    if not module.current_scene_graph or time.time() - module.current_scene_graph.last_update > 30:
        module.refresh_scene_graph()
    
    # Get spatial enhancements
    enhancement_request = {
        'type': 'AI_ENHANCEMENT_REQUEST',
        'data': {
            'user_request': user_request,
            'scene_data': scene_data or {}
        }
    }
    
    result = module.process_mcp_message(enhancement_request)
    
    if result.get('success'):
        return result
    else:
        # Fallback response
        return {
            'success': True,
            'semantic_context': {'relevant_objects': []},
            'ai_enhancements': {},
            'spatial_intelligence_applied': False,
            'fallback_used': True
        }

# Integration hooks for your existing execute_request function
def add_spatial_intelligence_to_request_execution():
    """Add spatial intelligence hooks to your existing request execution"""
    print("🌌 Spatial Intelligence integration hooks:")
    print("1. Call enhance_ai_request_with_spatial_context() before AI processing")
    print("2. Use query_spatial_intelligence() for specific spatial queries")
    print("3. Initialize with initialize_spatial_intelligence(llammy_core)")
    print("4. Scene graphs auto-refresh every 30 seconds or on demand")

# =============================================================================
# EXPORT INTERFACE
# =============================================================================

__all__ = [
    'SpatialIntelligenceMCP',
    'SpatialSceneGraph', 
    'BlenderSceneAnalyzer',
    'SpatialQueryEngine',
    'get_spatial_intelligence_module',
    'initialize_spatial_intelligence',
    'query_spatial_intelligence',
    'enhance_ai_request_with_spatial_context'
]

print("🌌 Llammy Spatial Intelligence Module v1.0 loaded successfully!")
print("Features:")
print("  ✅ 3D Scene Graph generation from Blender scenes")
print("  ✅ Spatial relationship analysis (NEAR, ABOVE, BELOW, etc.)")
print("  ✅ Intelligent spatial queries (path checking, collision detection)")
print("  ✅ MCP integration with your existing architecture")
print("  ✅ AI enhancement hooks for spatial reasoning")
print("  ✅ Embedded model integration for validation and scoring")
print("  ✅ NetworkX export support for advanced graph analysis")
print("Integration: Call initialize_spatial_intelligence(llammy_core) to connect")