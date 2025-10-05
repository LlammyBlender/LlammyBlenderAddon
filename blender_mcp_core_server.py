# blender_mcp_core_server.py
"""
Core MCP Server for Blender Integration
Handles basic scene state, object operations, and real-time monitoring
"""

import asyncio
import json
import websockets
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

@dataclass
class BlenderObject:
    name: str
    type: str
    location: List[float]
    rotation: List[float] 
    scale: List[float]
    visible: bool
    selected: bool
    active: bool

@dataclass
class SceneState:
    frame_current: int
    frame_start: int
    frame_end: int
    objects: List[BlenderObject]
    active_object: Optional[str]
    selected_objects: List[str]
    render_engine: str
    timestamp: str

class BlenderMCPServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.blender_connection = None
        self.current_state = None
        self.operation_log = []
        self.clients = set()
        
    async def start_server(self):
        """Start the MCP server"""
        logging.info(f"Starting Blender MCP Server on {self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"🚀 Blender MCP Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever
    
    async def handle_client(self, websocket, path):
        """Handle incoming MCP client connections"""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                response = await self.process_mcp_request(json.loads(message))
                await websocket.send(json.dumps(response))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
    
    async def process_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming MCP requests"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "get_scene_state":
                result = await self.get_scene_state()
            elif method == "create_object":
                result = await self.create_object(params)
            elif method == "delete_object":
                result = await self.delete_object(params)
            elif method == "move_object":
                result = await self.move_object(params)
            elif method == "set_keyframe":
                result = await self.set_keyframe(params)
            elif method == "get_operation_history":
                result = await self.get_operation_history()
            elif method == "start_monitoring":
                result = await self.start_state_monitoring()
            else:
                raise ValueError(f"Unknown method: {method}")
                
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            return {
                "jsonrpc": "2.0", 
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
    
    async def get_scene_state(self) -> Dict[str, Any]:
        """Get current Blender scene state"""
        # This would communicate with Blender addon to get real state
        # For now, returning mock data to establish the protocol
        
        if not self.blender_connection:
            # Mock state for development
            mock_objects = [
                BlenderObject(
                    name="Cube",
                    type="MESH",
                    location=[0.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0],
                    scale=[1.0, 1.0, 1.0],
                    visible=True,
                    selected=True,
                    active=True
                ),
                BlenderObject(
                    name="Camera",
                    type="CAMERA", 
                    location=[7.4, -6.5, 4.4],
                    rotation=[1.1, 0.0, 0.8],
                    scale=[1.0, 1.0, 1.0],
                    visible=True,
                    selected=False,
                    active=False
                )
            ]
            
            self.current_state = SceneState(
                frame_current=1,
                frame_start=1,
                frame_end=250,
                objects=mock_objects,
                active_object="Cube",
                selected_objects=["Cube"],
                render_engine="CYCLES",
                timestamp=datetime.now().isoformat()
            )
        
        return asdict(self.current_state)
    
    async def create_object(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create new object in scene"""
        obj_type = params.get("type", "MESH")
        name = params.get("name", "NewObject")
        location = params.get("location", [0.0, 0.0, 0.0])
        
        # Log operation for dataset
        operation = {
            "method": "create_object",
            "params": params,
            "timestamp": datetime.now().isoformat(),
            "scene_state_before": asdict(self.current_state) if self.current_state else None
        }
        
        # TODO: Send command to Blender addon
        # For now, simulate object creation
        new_object = BlenderObject(
            name=name,
            type=obj_type,
            location=location,
            rotation=[0.0, 0.0, 0.0],
            scale=[1.0, 1.0, 1.0],
            visible=True,
            selected=True,
            active=True
        )
        
        if self.current_state:
            self.current_state.objects.append(new_object)
            self.current_state.active_object = name
            self.current_state.selected_objects = [name]
        
        operation["scene_state_after"] = asdict(self.current_state) if self.current_state else None
        self.operation_log.append(operation)
        
        # Notify all connected clients of state change
        await self.broadcast_state_change()
        
        return {
            "success": True,
            "object_name": name,
            "message": f"Created {obj_type} object '{name}'"
        }
    
    async def delete_object(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Delete object from scene"""
        name = params.get("name")
        if not name:
            raise ValueError("Object name required")
        
        # Log operation
        operation = {
            "method": "delete_object",
            "params": params,
            "timestamp": datetime.now().isoformat(),
            "scene_state_before": asdict(self.current_state) if self.current_state else None
        }
        
        # TODO: Send command to Blender addon
        # Simulate deletion
        if self.current_state:
            self.current_state.objects = [
                obj for obj in self.current_state.objects 
                if obj.name != name
            ]
            if self.current_state.active_object == name:
                self.current_state.active_object = None
            if name in self.current_state.selected_objects:
                self.current_state.selected_objects.remove(name)
        
        operation["scene_state_after"] = asdict(self.current_state) if self.current_state else None
        self.operation_log.append(operation)
        
        await self.broadcast_state_change()
        
        return {
            "success": True,
            "message": f"Deleted object '{name}'"
        }
    
    async def move_object(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Move object to new location"""
        name = params.get("name")
        location = params.get("location")
        
        if not name or not location:
            raise ValueError("Object name and location required")
        
        # Log operation
        operation = {
            "method": "move_object", 
            "params": params,
            "timestamp": datetime.now().isoformat(),
            "scene_state_before": asdict(self.current_state) if self.current_state else None
        }
        
        # TODO: Send command to Blender addon
        # Simulate movement
        if self.current_state:
            for obj in self.current_state.objects:
                if obj.name == name:
                    obj.location = location
                    break
        
        operation["scene_state_after"] = asdict(self.current_state) if self.current_state else None
        self.operation_log.append(operation)
        
        await self.broadcast_state_change()
        
        return {
            "success": True,
            "message": f"Moved '{name}' to {location}"
        }
    
    async def set_keyframe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set keyframe for object property"""
        name = params.get("name")
        property_type = params.get("property", "location")
        frame = params.get("frame", 1)
        
        if not name:
            raise ValueError("Object name required")
        
        # Log operation
        operation = {
            "method": "set_keyframe",
            "params": params,
            "timestamp": datetime.now().isoformat(),
            "scene_state_before": asdict(self.current_state) if self.current_state else None
        }
        
        # TODO: Send command to Blender addon
        # For now just log the keyframe operation
        
        operation["scene_state_after"] = asdict(self.current_state) if self.current_state else None
        self.operation_log.append(operation)
        
        return {
            "success": True,
            "message": f"Set {property_type} keyframe for '{name}' at frame {frame}"
        }
    
    async def get_operation_history(self) -> Dict[str, Any]:
        """Get operation history for dataset building"""
        return {
            "operations": self.operation_log,
            "total_operations": len(self.operation_log)
        }
    
    async def start_state_monitoring(self) -> Dict[str, Any]:
        """Start real-time state monitoring"""
        # TODO: Implement real-time monitoring loop
        return {
            "monitoring": True,
            "message": "State monitoring started"
        }
    
    async def broadcast_state_change(self):
        """Broadcast state changes to all connected clients"""
        if self.clients:
            state_update = {
                "type": "state_change",
                "data": asdict(self.current_state) if self.current_state else None
            }
            
            # Send to all connected AI clients
            disconnected = set()
            for client in self.clients:
                try:
                    await client.send(json.dumps(state_update))
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(client)
            
            # Clean up disconnected clients
            self.clients -= disconnected

# Development server runner
async def main():
    logging.basicConfig(level=logging.INFO)
    server = BlenderMCPServer()
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main())