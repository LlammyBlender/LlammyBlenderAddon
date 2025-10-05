# =============================================================================
# LLAMMY BLENDER ADDON - Complete __init__.py
# Enhanced modular dual AI for end-to-end animation with MCP architecture
# =============================================================================

import bpy
import os
import time
import sys
import requests
import json
import asyncio
import websockets
import threading
import queue
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import StringProperty, BoolProperty, IntProperty
import logging

# =============================================================================
# ADDON INFORMATION
# =============================================================================

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
# DEBUG AND LOGGING SYSTEM
# =============================================================================

class DebugManager:
    """Centralized debug and logging system"""
    
    def __init__(self):
        self.logs = []
        self.max_logs = 100
        
    def log(self, level: str, message: str):
        """Add log entry with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.logs.append(log_entry)
        
        # Keep log size in check
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

debug = DebugManager()

# =============================================================================
# CORE THREAD-SAFE EXECUTION (THE 'MISSING PIECE')
# =============================================================================

# This queue stores operations from the background threads to be executed
# on Blender's main thread, ensuring thread safety.
operation_queue = queue.Queue()

# Timer to process the queue on Blender's main thread
def process_operations():
    try:
        # Get all available operations from the queue
        while True:
            operation = operation_queue.get_nowait()
            operation['callback'](**operation['params'])
    except queue.Empty:
        pass
    # Re-register the timer to run again, creating a continuous loop
    return 0.1

# Register the timer to start processing the queue
bpy.app.timers.register(process_operations)

# =============================================================================
# MCP BRIDGE CORE CLASS
# This class handles WebSocket communication and command dispatching
# =============================================================================

class MCPBlenderBridge:
    def __init__(self):
        self.server_host = "localhost"
        self.server_port = 8765
        self.websocket = None
        self.connected = False
        self.listener_thread = None
        self.should_monitor = False

    async def connect(self):
        uri = f"ws://{self.server_host}:{self.server_port}"
        try:
            self.websocket = await websockets.connect(uri)
            self.connected = True
            debug.log("INFO", f"✅ Successfully connected to MCP server at {uri}")
            
            # Start a separate listener thread
            self.listener_thread = threading.Thread(target=self.run_listener_loop, daemon=True)
            self.listener_thread.start()

        except Exception as e:
            debug.log("ERROR", f"❌ Failed to connect to MCP server: {e}")
            self.connected = False

    def run_listener_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.listen_for_commands())

    async def listen_for_commands(self):
        try:
            while self.connected:
                message_json = await self.websocket.recv()
                message = json.loads(message_json)
                if message.get('type') == 'request':
                    self.dispatch_command(message.get('data', {}))
                
        except websockets.exceptions.ConnectionClosed:
            debug.log("WARNING", "❌ Connection to MCP server closed.")
            self.connected = False
        except Exception as e:
            debug.log("ERROR", f"❌ An error occurred in the listener: {e}")

    def dispatch_command(self, data: Dict[str, Any]):
        """Queues a command for execution on Blender's main thread."""
        method_name = data.get('method')
        params = data.get('params', {})
        
        # Add the operation to the queue for the main thread to execute
        operation_queue.put_nowait({
            'callback': self.execute_mcp_command,
            'params': {'method': method_name, 'params': params}
        })
        debug.log("INFO", f"➡️ Queued command for main thread: {method_name}")

    def execute_mcp_command(self, method: str, params: Dict[str, Any]):
        """Directly executes the command using Blender's API."""
        try:
            if method == "create_object":
                obj_type = params.get('type', 'MESH')
                if obj_type == "MESH":
                    bpy.ops.mesh.primitive_cube_add(location=params.get('location', (0,0,0)))
                    debug.log("INFO", "✅ Created new cube.")
                # Add more object types as needed

            elif method == "select_object":
                obj_name = params.get('name')
                bpy.ops.object.select_all(action='DESELECT')
                if obj_name in bpy.data.objects:
                    bpy.data.objects[obj_name].select_set(True)
                    bpy.context.view_layer.objects.active = bpy.data.objects[obj_name]
                    debug.log("INFO", f"✅ Selected object: {obj_name}")
                else:
                    debug.log("WARNING", f"❌ Object not found for selection: {obj_name}")

            elif method == "delete_object":
                obj_name = params.get('name')
                if obj_name in bpy.data.objects:
                    bpy.data.objects[obj_name].select_set(True)
                    bpy.ops.object.delete(use_global=False)
                    debug.log("INFO", f"✅ Deleted object: {obj_name}")
                else:
                    debug.log("WARNING", f"❌ Object not found for deletion: {obj_name}")
            
            # Add more methods here, like 'move_object', 'set_material', etc.

            else:
                debug.log("WARNING", f"⚠️ Unknown command received: {method}")

        except Exception as e:
            debug.log("ERROR", f"❌ Failed to execute command '{method}': {e}")
    
    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            debug.log("INFO", "🛑 Disconnected from MCP server.")

# Global MCP Bridge instance
mcp_bridge = MCPBlenderBridge()

# =============================================================================
# BLENDER UI AND OPERATORS
# =============================================================================

class MCP_PT_MainPanel(Panel):
    # Your existing panel code here...
    bl_label = "MCP AI Bridge"
    bl_idname = "MCP_PT_MAIN_PANEL"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'MCP AI'

    def draw(self, context):
        layout = self.layout
        
        # Connection status
        box = layout.box()
        status_row = box.row()
        status_row.label(text="Connection Status:")
        
        if mcp_bridge.connected:
            status_row.label(text="Connected", icon='DOT')
            layout.operator("mcp.disconnect_server")
        else:
            status_row.label(text="Disconnected", icon='RADIOBUT_OFF')
            layout.operator("mcp.connect_server")
            
        # Logging area
        box = layout.box()
        box.label(text="Log:")
        for log_entry in debug.logs:
            box.label(text=log_entry)

class MCP_OT_ConnectServer(Operator):
    bl_idname = "mcp.connect_server"
    bl_label = "Connect to MCP Server"
    bl_description = "Connect to the external MCP server"
    
    def execute(self, context):
        # Start the connection in a new thread
        thread = threading.Thread(target=asyncio.run, args=(mcp_bridge.connect(),))
        thread.daemon = True
        thread.start()
        return {'FINISHED'}

class MCP_OT_DisconnectServer(Operator):
    bl_idname = "mcp.disconnect_server"
    bl_label = "Disconnect from MCP Server"
    bl_description = "Disconnect from the external MCP server"
    
    def execute(self, context):
        thread = threading.Thread(target=asyncio.run, args=(mcp_bridge.disconnect(),))
        thread.daemon = True
        thread.start()
        return {'FINISHED'}

# ... (other existing operators go here)

# =============================================================================
# REGISTRATION
# =============================================================================

classes = [
    MCP_PT_MainPanel,
    MCP_OT_ConnectServer,
    MCP_OT_DisconnectServer,
    # Add other operator classes here
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # You can also automatically start the connection on registration if you prefer
    # thread = threading.Thread(target=asyncio.run, args=(mcp_bridge.connect(),))
    # thread.daemon = True
    # thread.start()
    
    debug.log("INFO", "🚀 Llammy MCP AI Bridge addon registered")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    # Clean up the connection when the addon is disabled
    if mcp_bridge.connected:
        try:
            thread = threading.Thread(target=asyncio.run, args=(mcp_bridge.disconnect(),))
            thread.daemon = True
            thread.start()
            thread.join(timeout=2) # Give it a moment to close
        except Exception as e:
            debug.log("ERROR", f"Failed to gracefully unregister: {e}")

    debug.log("INFO", "👋 Llammy MCP AI Bridge addon unregistered")

if __name__ == "__main__":
    register()
