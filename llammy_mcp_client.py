# llammy_mcp_client.py
"""
Llammy (Llama3.2:3B) MCP Client
Connects to MCP servers and orchestrates AI-driven Blender operations
"""

import asyncio
import json
import websockets
import aiohttp
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import re

@dataclass
class AIDecision:
    action: str
    confidence: float
    reasoning: str
    parameters: Dict[str, Any]
    timestamp: str

class LlammyMCPClient:
    def __init__(self,
                 ollama_host="localhost",
                 ollama_port=11434,
                 mcp_host="localhost",
                 mcp_port=8765):
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.mcp_host = mcp_host
        self.mcp_port = mcp_port
        
        self.mcp_connection = None
        self.current_scene_state = None
        self.operation_history = []
        self.ai_decisions = []
        
        self.system_prompt = self._build_system_prompt()
        
    async def start_client(self):
        """Start the Llammy MCP client"""
        logging.info("🧠 Starting Llammy MCP Client...")
        
        # Connect to MCP server
        await self.connect_to_mcp()
        
        # Get initial scene state
        await self.refresh_scene_state()
        
        print("🚀 Llammy is online and connected!")
        
        # Start decision loop
        await self.decision_loop()
    
    async def connect_to_mcp(self):
        """Connect to Blender MCP server"""
        try:
            uri = f"ws://{self.mcp_host}:{self.mcp_port}"
            self.mcp_connection = await websockets.connect(uri)
            print(f"✅ Connected to Blender MCP: {uri}")
        except Exception as e:
            print(f"❌ Failed to connect to MCP server: {e}")
            raise
    
    async def refresh_scene_state(self):
        """Get current scene state from MCP server"""
        if not self.mcp_connection:
            return
            
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "get_scene_state",
            "params": {}
        }
        
        try:
            await self.mcp_connection.send(json.dumps(request))
            response = await self.mcp_connection.recv()
            result = json.loads(response)
            
            if "result" in result:
                self.current_scene_state = result["result"]
                print(f"📊 Scene state updated: {len(self.current_scene_state.get('objects', []))} objects")
            else:
                print(f"⚠️ MCP Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Failed to get scene state: {e}")
    
    async def ask_llammy(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query Llammy (Llama3.2:3B) via Ollama"""
        
        # Build full context prompt
        full_prompt = f"{self.system_prompt}\n\n"
        
        if self.current_scene_state:
            full_prompt += f"CURRENT SCENE STATE:\n{json.dumps(self.current_scene_state, indent=2)}\n\n"
        
        if context:
            full_prompt += f"ADDITIONAL CONTEXT:\n{json.dumps(context, indent=2)}\n\n"
            
        if self.operation_history:
            recent_ops = self.operation_history[-5:]  # Last 5 operations
            full_prompt += f"RECENT OPERATIONS:\n{json.dumps(recent_ops, indent=2)}\n\n"
        
        full_prompt += f"USER REQUEST: {prompt}\n\nRespond with structured JSON as specified:"
        
        # Send to Ollama
        try:
            async with aiohttp.ClientSession() as session:
                ollama_url = f"http://{self.ollama_host}:{self.ollama_port}/api/generate"
                
                payload = {
                    "model": "llama3.2:3b",
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2048
                    }
                }
                
                async with session.post(ollama_url, json=payload) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        response_text = result["response"]
                        
                        # Parse JSON response
                        try:
                            # Extract JSON from response (handle markdown formatting)
                            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                            if json_match:
                                ai_response = json.loads(json_match.group())
                                return ai_response
                            else:
                                # Fallback parsing
                                return {"analysis": response_text, "actions": []}
                                
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Failed to parse Llammy response as JSON: {e}")
                            return {"analysis": response_text, "actions": []}
                    else:
                        print(f"❌ Ollama request failed: {resp.status}")
                        return {"error": f"Ollama request failed: {resp.status}"}
                        
        except Exception as e:
            print(f"❌ Error querying Llammy: {e}")
            return {"error": str(e)}
    
    async def execute_ai_decision(self, decision: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute AI decision by calling MCP methods"""
        if not self.mcp_connection:
            return [{"error": "No MCP connection"}]
            
        results = []
        actions = decision.get("actions", [])
        
        for i, action in enumerate(actions):
            method = action.get("method")
            params = action.get("params", {})
            expected = action.get("expected_outcome", "Unknown")
            
            print(f"🎬 Executing action {i+1}/{len(actions)}: {method}")
            print(f"   Expected: {expected}")
            
            # Send MCP request
            request = {
                "jsonrpc": "2.0",
                "id": i + 1,
                "method": method,
                "params": params
            }
            
            try:
                await self.mcp_connection.send(json.dumps(request))
                response = await self.mcp_connection.recv()
                result = json.loads(response)
                
                if "result" in result:
                    print(f"✅ Action succeeded: {result['result']}")
                    results.append({
                        "action": action,
                        "result": result["result"],
                        "success": True
                    })
                else:
                    error = result.get("error", {})
                    print(f"❌ Action failed: {error.get('message', 'Unknown error')}")
                    results.append({
                        "action": action,
                        "error": error,
                        "success": False
                    })
                    
                # Small delay between operations
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"❌ MCP request failed: {e}")
                results.append({
                    "action": action,
                    "error": str(e),
                    "success": False
                })
        
        # Record decision and results for learning
        decision_record = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "results": results,
            "scene_state_before": self.current_scene_state
        }
        
        self.ai_decisions.append(decision_record)
        
        # Refresh scene state after operations
        await self.refresh_scene_state()
        
        decision_record["scene_state_after"] = self.current_scene_state
        
        return results
    
    async def decision_loop(self):
        """Main AI decision making loop"""
        print("🧠 Starting AI decision loop...")
        
        while True:
            try:
                # Check for user input or autonomous decisions
                user_input = input("\n💭 What should I do? (or 'auto' for autonomous mode): ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'auto':
                    # Autonomous mode - AI decides what to do
                    prompt = "Analyze the current scene and suggest improvements, animations, or creative enhancements. Be bold and creative!"
                else:
                    prompt = user_input
                
                if prompt:
                    if user_input.lower() != 'multi':  # Multi-model already processed
                        print(f"🤔 Asking {self.current_model}: {prompt}")
                        
                        # Get AI decision
                        decision = await self.ask_llammy(prompt)
                    
                    if "error" in decision:
                        print(f"❌ Llammy error: {decision['error']}")
                        continue
                    
                    # Display AI reasoning
                    print(f"\n🧠 Llammy's Analysis:")
                    print(f"   {decision.get('analysis', 'No analysis provided')}")
                    print(f"\n🎯 Decision: {decision.get('decision', 'No decision')}")
                    print(f"🤔 Reasoning: {decision.get('reasoning', 'No reasoning')}")
                    print(f"📊 Confidence: {decision.get('confidence', 0.0)}")
                    
                    # Show planned actions
                    actions = decision.get("actions", [])
                    if actions:
                        print(f"\n🎬 Planned Actions ({len(actions)}):")
                        for i, action in enumerate(actions, 1):
                            print(f"   {i}. {action.get('method', 'unknown')} - {action.get('expected_outcome', 'unknown outcome')}")
                    
                    # Ask for confirmation
                    if actions:
                        confirm = input("\n🚀 Execute these actions? (y/n/auto): ").strip().lower()
                        if confirm in ['y', 'yes', 'auto']:
                            results = await self.execute_ai_decision(decision)
                            
                            # Show results summary
                            successful = sum(1 for r in results if r.get("success", False))
                            print(f"\n📈 Results: {successful}/{len(results)} actions successful")
                            
                            if successful < len(results):
                                # Some failures - ask AI to debug
                                debug_prompt = f"Some operations failed. Results: {json.dumps(results, indent=2)}. How can we fix this?"
                                print("\n🔧 Asking Llammy to debug failures...")
                                debug_decision = await self.ask_llammy(debug_prompt)
                                print(f"🛠️ Debug suggestion: {debug_decision.get('analysis', 'No debug info')}")
                        else:
                            print("⏸️ Actions cancelled")
                    else:
                        print("⚠️ No actions planned")
                
            except KeyboardInterrupt:
                print("\n👋 Stopping decision loop...")
                break
            except Exception as e:
                print(f"❌ Decision loop error: {e}")
                await asyncio.sleep(1)
    
    async def save_learning_data(self, filename: str = None):
        """Save all decisions and operations for dataset building"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llammy_learning_data_{timestamp}.json"
        
        learning_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_decisions": len(self.ai_decisions),
                "total_operations": len(self.operation_history)
            },
            "ai_decisions": self.ai_decisions,
            "operation_history": self.operation_history,
            "final_scene_state": self.current_scene_state
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(learning_data, f, indent=2)
            print(f"💾 Learning data saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save learning data: {e}")

# Main runner
async def main():
    logging.basicConfig(level=logging.INFO)
    
    print("🧠 Starting Llammy MCP Client...")
    print("   Make sure:")
    print("   1. Ollama is running with llama3.2:3b model")
    print("   2. Blender MCP server is running")
    print("   3. Blender addon is connected")
    
    client = LlammyMCPClient()
    
    try:
        await client.start_client()
    except KeyboardInterrupt:
        print("\n👋 Shutting down Llammy...")
    finally:
        # Save learning data before exit
        await client.save_learning_data()
        if client.mcp_connection:
            await client.mcp_connection.close()

if __name__ == "__main__":
    asyncio.run(main())
        
    def _build_system_prompt(self) -> str:
        return """You are Llammy, an AI specialized in Blender 3D animation and scene management. 

You
