# =============================================================================
# MCP DEBUG/REGENERATION SYSTEM WITH HUMAN-IN-THE-LOOP
# Enhanced MCP architecture with failure detection, debugging, and human oversight
# =============================================================================

import asyncio
import json
import time
import traceback
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

class FailureType(Enum):
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    BLENDER_API_ERROR = "blender_api_error"
    LOGIC_ERROR = "logic_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_FAILURE = "validation_failure"

class ActionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    HUMAN_REVIEW = "human_review"
    REGENERATING = "regenerating"

@dataclass
class FailureContext:
    failure_type: FailureType
    error_message: str
    stack_trace: str
    scene_state_before: Dict[str, Any]
    attempted_action: Dict[str, Any]
    timestamp: float
    human_feedback: Optional[str] = None
    regeneration_attempts: int = 0

@dataclass
class ActionResult:
    action_id: str
    status: ActionStatus
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    execution_time: float
    validation_passed: bool
    failure_context: Optional[FailureContext] = None
class DualModelOrchestrator:
    def __init__(self):
        self.qwen_debugger = QwenDebugger()      # Handles failures, validation
        self.gemma_harvester = GemmaHarvester()  # Handles data collection, analysis
        self.active_model = None
        self.coordination_queue = []

class EmbeddedDebugger:
    """Qwen-based debugging system for MCP operations"""
    
    def __init__(self, ollama_client):
        self.ollama_client = ollama_client
        self.debug_history = []
        self.regeneration_patterns = {}
        
    async def analyze_failure(self, failure: FailureContext) -> Dict[str, Any]:
        """Analyze failure and suggest fixes using embedded Qwen model"""
        
        debug_prompt = f"""
BLENDER MCP OPERATION FAILURE ANALYSIS:

Failure Type: {failure.failure_type.value}
Error Message: {failure.error_message}
Stack Trace: {failure.stack_trace}

Scene State Before: {json.dumps(failure.scene_state_before, indent=2)}
Attempted Action: {json.dumps(failure.attempted_action, indent=2)}

Previous Regeneration Attempts: {failure.regeneration_attempts}

Analyze this failure and provide:
1. Root cause identification
2. Specific fix suggestions
3. Alternative approaches
4. Prevention strategies

Respond in JSON format:
{{
    "root_cause": "detailed analysis of what went wrong",
    "confidence": 0.0-1.0,
    "fix_suggestions": [
        {{
            "approach": "description",
            "changes_needed": ["specific changes"],
            "risk_level": "low/medium/high",
            "success_probability": 0.0-1.0
        }}
    ],
    "alternative_approaches": [
        {{
            "method": "alternative method",
            "description": "why this might work better",
            "implementation": "how to implement"
        }}
    ],
    "prevention_strategies": ["future prevention methods"],
    "requires_human_review": true/false,
    "human_review_reason": "why human input is needed"
}}
"""
        
        try:
            response = await self.ollama_client.generate("qwen3:0.6b", debug_prompt)
            
            # Parse JSON response
            debug_analysis = self._extract_json(response)
            
            # Store for learning
            self.debug_history.append({
                'failure': asdict(failure),
                'analysis': debug_analysis,
                'timestamp': time.time()
            })
            
            return debug_analysis
            
        except Exception as e:
            logging.error(f"Debug analysis failed: {e}")
            return {
                "root_cause": f"Debug analysis failed: {str(e)}",
                "confidence": 0.1,
                "requires_human_review": True,
                "human_review_reason": "Automated debugging failed"
            }
    
    async def generate_fix(self, failure: FailureContext, analysis: Dict[str, Any], human_feedback: str = None) -> Dict[str, Any]:
        """Generate corrected action based on failure analysis"""
        
        fix_prompt = f"""
GENERATE CORRECTED BLENDER MCP ACTION:

Original Failed Action: {json.dumps(failure.attempted_action, indent=2)}
Failure Analysis: {json.dumps(analysis, indent=2)}
Scene State: {json.dumps(failure.scene_state_before, indent=2)}

Human Feedback: {human_feedback or "None provided"}
Regeneration Attempt: {failure.regeneration_attempts + 1}

Generate a corrected MCP action that addresses the root cause.
Consider:
1. API compatibility with Blender 4.4+
2. Scene state constraints
3. Error prevention
4. Human feedback integration

Respond with corrected action in JSON format:
{{
    "method": "mcp_method_name",
    "params": {{
        "parameter1": "value",
        "parameter2": "value"
    }},
    "validation_checks": [
        "check1: description",
        "check2: description"
    ],
    "expected_outcome": "what should happen",
    "rollback_plan": "how to undo if this fails",
    "confidence": 0.0-1.0,
    "changes_made": ["list of changes from original"],
    "human_verification_needed": true/false
}}
"""
        
        try:
            response = await self.ollama_client.generate("qwen3:0.6b", fix_prompt)
            corrected_action = self._extract_json(response)
            
            # Learn from regeneration patterns
            self._learn_regeneration_pattern(failure, corrected_action)
            
            return corrected_action
            
        except Exception as e:
            logging.error(f"Fix generation failed: {e}")
            return {
                "error": f"Fix generation failed: {str(e)}",
                "human_verification_needed": True
            }
    
    def _extract_json(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from model response"""
        import re
        
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: create structured response from text
        return {
            "raw_response": response_text,
            "requires_human_review": True,
            "human_review_reason": "Could not parse AI response"
        }
    
    def _learn_regeneration_pattern(self, failure: FailureContext, fix: Dict[str, Any]):
        """Learn from successful regeneration patterns"""
        pattern_key = f"{failure.failure_type.value}:{failure.attempted_action.get('method', 'unknown')}"
        
        if pattern_key not in self.regeneration_patterns:
            self.regeneration_patterns[pattern_key] = []
        
        self.regeneration_patterns[pattern_key].append({
            'original_error': failure.error_message,
            'fix_applied': fix,
            'timestamp': time.time()
        })
        
        # Keep only recent patterns (last 10)
        self.regeneration_patterns[pattern_key] = self.regeneration_patterns[pattern_key][-10:]

class HumanReviewInterface:
    """Human-in-the-loop review system"""
    
    def __init__(self):
        self.pending_reviews = {}
        self.review_callbacks = {}
        
    async def request_human_review(self, 
                                 action_id: str,
                                 failure: FailureContext,
                                 analysis: Dict[str, Any],
                                 proposed_fix: Dict[str, Any]) -> str:
        """Request human review and wait for response"""
        
        review_request = {
            'action_id': action_id,
            'failure_summary': {
                'type': failure.failure_type.value,
                'error': failure.error_message,
                'attempts': failure.regeneration_attempts
            },
            'ai_analysis': analysis.get('root_cause', 'No analysis'),
            'proposed_fix': proposed_fix,
            'options': [
                'approve_fix',
                'modify_fix', 
                'try_alternative',
                'abort_action',
                'manual_intervention'
            ]
        }
        
        self.pending_reviews[action_id] = review_request
        
        # In a real implementation, this would trigger UI notification
        print(f"\n🔍 HUMAN REVIEW REQUESTED - Action ID: {action_id}")
        print(f"Failure: {failure.failure_type.value} - {failure.error_message}")
        print(f"AI Analysis: {analysis.get('root_cause', 'Unknown')}")
        print(f"Proposed Fix: {proposed_fix.get('method', 'Unknown')} with confidence {proposed_fix.get('confidence', 0.0)}")
        
        # Simulate human input (in real system, this would come from UI)
        print("\nOptions:")
        print("1. approve_fix - Use AI's proposed fix")
        print("2. modify_fix - Provide modifications") 
        print("3. try_alternative - Try different approach")
        print("4. abort_action - Cancel this operation")
        print("5. manual_intervention - Switch to manual mode")
        
        choice = input("Human decision (1-5): ").strip()
        
        decision_map = {
            '1': 'approve_fix',
            '2': 'modify_fix',
            '3': 'try_alternative', 
            '4': 'abort_action',
            '5': 'manual_intervention'
        }
        
        decision = decision_map.get(choice, 'abort_action')
        
        # Get additional feedback if needed
        feedback = ""
        if decision in ['modify_fix', 'try_alternative']:
            feedback = input("Additional instructions: ").strip()
        
        del self.pending_reviews[action_id]
        
        return json.dumps({
            'decision': decision,
            'feedback': feedback,
            'timestamp': time.time()
        })

class MCPActionValidator:
    """Pre-execution validation system"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        
    async def validate_action(self, action: Dict[str, Any], scene_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate action before execution"""
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'suggestions': []
        }
        
        method = action.get('method', '')
        params = action.get('params', {})
        
        # Method validation
        if method not in self.validation_rules:
            validation_results['warnings'].append(f"Unknown method: {method}")
        
        # Parameter validation
        if method == 'create_object':
            if 'name' not in params:
                validation_results['errors'].append("Object name required")
                validation_results['valid'] = False
            
            # Check for name conflicts
            existing_objects = [obj['name'] for obj in scene_state.get('objects', [])]
            if params.get('name') in existing_objects:
                validation_results['warnings'].append(f"Object '{params.get('name')}' already exists")
                validation_results['suggestions'].append(f"Use unique name like '{params.get('name')}_001'")
        
        elif method == 'move_object':
            obj_name = params.get('name')
            if not obj_name:
                validation_results['errors'].append("Object name required for move")
                validation_results['valid'] = False
            else:
                existing_objects = [obj['name'] for obj in scene_state.get('objects', [])]
                if obj_name not in existing_objects:
                    validation_results['errors'].append(f"Object '{obj_name}' not found")
                    validation_results['valid'] = False
        
        elif method == 'set_keyframe':
            frame = params.get('frame')
            if frame is not None and frame < 1:
                validation_results['errors'].append("Frame number must be >= 1")
                validation_results['valid'] = False
        
        return validation_results
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for different methods"""
        return {
            'create_object': {
                'required_params': ['name'],
                'optional_params': ['type', 'location', 'rotation', 'scale'],
                'param_types': {
                    'name': str,
                    'type': str,
                    'location': list,
                    'rotation': list,
                    'scale': list
                }
            },
            'move_object': {
                'required_params': ['name', 'location'],
                'param_types': {
                    'name': str,
                    'location': list
                }
            },
            'delete_object': {
                'required_params': ['name'],
                'param_types': {
                    'name': str
                }
            },
            'set_keyframe': {
                'required_params': ['name'],
                'optional_params': ['property', 'frame'],
                'param_types': {
                    'name': str,
                    'property': str,
                    'frame': int
                }
            }
        }

class EnhancedMCPExecutor:
    """Enhanced MCP executor with debug/regen capabilities"""
    
    def __init__(self, ollama_client, mcp_connection):
        self.ollama_client = ollama_client
        self.mcp_connection = mcp_connection
        self.debugger = EmbeddedDebugger(ollama_client)
        self.validator = MCPActionValidator()
        self.human_review = HumanReviewInterface()
        
        self.execution_history = []
        self.active_actions = {}
        
    async def execute_action_with_recovery(self, action: Dict[str, Any], scene_state: Dict[str, Any]) -> ActionResult:
        """Execute action with full debug/recovery pipeline"""
        
        action_id = f"action_{int(time.time())}_{len(self.execution_history)}"
        start_time = time.time()
        
        # Phase 1: Pre-execution validation
        validation = await self.validator.validate_action(action, scene_state)
        
        if not validation['valid']:
            return ActionResult(
                action_id=action_id,
                status=ActionStatus.FAILED,
                result=None,
                error=f"Validation failed: {validation['errors']}",
                execution_time=time.time() - start_time,
                validation_passed=False
            )
        
        # Show validation warnings
        if validation['warnings']:
            print(f"⚠️ Validation warnings: {validation['warnings']}")
        
        # Phase 2: Execute with monitoring
        result = await self._execute_with_monitoring(action_id, action, scene_state, start_time)
        
        # Phase 3: Handle failures with recovery
        if result.status == ActionStatus.FAILED and result.failure_context:
            result = await self._handle_failure_with_recovery(result)
        
        # Phase 4: Store results
        self.execution_history.append(result)
        
        return result
    
    async def _execute_with_monitoring(self, action_id: str, action: Dict[str, Any], scene_state: Dict[str, Any], start_time: float) -> ActionResult:
        """Execute action with failure monitoring"""
        
        try:
            self.active_actions[action_id] = {
                'action': action,
                'start_time': start_time,
                'status': ActionStatus.EXECUTING
            }
            
            # Send MCP request
            request = {
                "jsonrpc": "2.0",
                "id": action_id,
                "method": action.get('method'),
                "params": action.get('params', {})
            }
            
            await self.mcp_connection.send(json.dumps(request))
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(
                    self.mcp_connection.recv(), 
                    timeout=30.0
                )
                result_data = json.loads(response)
                
                if "result" in result_data:
                    # Success
                    del self.active_actions[action_id]
                    return ActionResult(
                        action_id=action_id,
                        status=ActionStatus.SUCCESS,
                        result=result_data["result"],
                        error=None,
                        execution_time=time.time() - start_time,
                        validation_passed=True
                    )
                else:
                    # MCP error
                    error = result_data.get("error", {})
                    failure_context = FailureContext(
                        failure_type=FailureType.BLENDER_API_ERROR,
                        error_message=error.get("message", "Unknown MCP error"),
                        stack_trace="",
                        scene_state_before=scene_state,
                        attempted_action=action,
                        timestamp=time.time()
                    )
                    
                    return ActionResult(
                        action_id=action_id,
                        status=ActionStatus.FAILED,
                        result=None,
                        error=error.get("message", "Unknown MCP error"),
                        execution_time=time.time() - start_time,
                        validation_passed=True,
                        failure_context=failure_context
                    )
                    
            except asyncio.TimeoutError:
                failure_context = FailureContext(
                    failure_type=FailureType.TIMEOUT_ERROR,
                    error_message="Action timed out after 30 seconds",
                    stack_trace="",
                    scene_state_before=scene_state,
                    attempted_action=action,
                    timestamp=time.time()
                )
                
                return ActionResult(
                    action_id=action_id,
                    status=ActionStatus.FAILED,
                    result=None,
                    error="Execution timeout",
                    execution_time=time.time() - start_time,
                    validation_passed=True,
                    failure_context=failure_context
                )
                
        except Exception as e:
            failure_context = FailureContext(
                failure_type=FailureType.RUNTIME_ERROR,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                scene_state_before=scene_state,
                attempted_action=action,
                timestamp=time.time()
            )
            
            return ActionResult(
                action_id=action_id,
                status=ActionStatus.FAILED,
                result=None,
                error=str(e),
                execution_time=time.time() - start_time,
                validation_passed=True,
                failure_context=failure_context
            )
        finally:
            if action_id in self.active_actions:
                del self.active_actions[action_id]
    
    async def _handle_failure_with_recovery(self, failed_result: ActionResult) -> ActionResult:
        """Handle failure with AI debugging and human oversight"""
        
        failure = failed_result.failure_context
        max_regeneration_attempts = 3
        
        while failure.regeneration_attempts < max_regeneration_attempts:
            print(f"\n🔧 Debugging failure (attempt {failure.regeneration_attempts + 1}/{max_regeneration_attempts})")
            
            # Step 1: AI analysis
            analysis = await self.debugger.analyze_failure(failure)
            print(f"🤖 AI Analysis: {analysis.get('root_cause', 'Unknown')}")
            
            # Step 2: Check if human review is needed
            if analysis.get('requires_human_review', False):
                print("👤 Human review required")
                failed_result.status = ActionStatus.HUMAN_REVIEW
                
                # Generate proposed fix
                proposed_fix = await self.debugger.generate_fix(failure, analysis)
                
                # Request human review
                human_response = await self.human_review.request_human_review(
                    failed_result.action_id,
                    failure,
                    analysis,
                    proposed_fix
                )
                
                human_data = json.loads(human_response)
                decision = human_data.get('decision')
                feedback = human_data.get('feedback', '')
                
                if decision == 'abort_action':
                    failed_result.status = ActionStatus.FAILED
                    failed_result.error += " (Aborted by human)"
                    break
                elif decision == 'manual_intervention':
                    failed_result.status = ActionStatus.HUMAN_REVIEW
                    failed_result.error += " (Manual intervention required)"
                    break
                elif decision in ['approve_fix', 'modify_fix', 'try_alternative']:
                    # Update failure context with human feedback
                    failure.human_feedback = feedback
                    
                    if decision == 'modify_fix' and feedback:
                        # Regenerate fix with human feedback
                        proposed_fix = await self.debugger.generate_fix(failure, analysis, feedback)
            else:
                # Auto-regenerate without human review
                proposed_fix = await self.debugger.generate_fix(failure, analysis)
            
            # Step 3: Attempt regeneration
            if proposed_fix.get('error'):
                print(f"❌ Fix generation failed: {proposed_fix['error']}")
                break
            
            failure.regeneration_attempts += 1
            failed_result.status = ActionStatus.REGENERATING
            
            print(f"🔄 Attempting regenerated action (confidence: {proposed_fix.get('confidence', 0.0)})")
            
            # Execute corrected action
            corrected_result = await self._execute_with_monitoring(
                f"{failed_result.action_id}_regen_{failure.regeneration_attempts}",
                proposed_fix,
                failure.scene_state_before,
                time.time()
            )
            
            if corrected_result.status == ActionStatus.SUCCESS:
                print("✅ Regeneration successful!")
                failed_result.status = ActionStatus.SUCCESS
                failed_result.result = corrected_result.result
                failed_result.error = None
                break
            else:
                print(f"❌ Regeneration failed: {corrected_result.error}")
                # Update failure context for next attempt
                failure.error_message = corrected_result.error
                if corrected_result.failure_context:
                    failure.failure_type = corrected_result.failure_context.failure_type
        
        return failed_result
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total = len(self.execution_history)
        if total == 0:
            return {'total': 0, 'success_rate': 0.0}
        
        successful = sum(1 for r in self.execution_history if r.status == ActionStatus.SUCCESS)
        failed = sum(1 for r in self.execution_history if r.status == ActionStatus.FAILED)
        human_reviews = sum(1 for r in self.execution_history if r.status == ActionStatus.HUMAN_REVIEW)
        
        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'human_reviews': human_reviews,
            'success_rate': successful / total,
            'avg_execution_time': sum(r.execution_time for r in self.execution_history) / total,
            'regeneration_patterns': len(self.debugger.regeneration_patterns)
        }

# Example usage integration with existing Llammy client
class EnhancedLlammyMCPClient:
    """Enhanced Llammy client with debug/regen capabilities"""
    
    def __init__(self, ollama_host="localhost", ollama_port=11434, mcp_host="localhost", mcp_port=8765):
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.mcp_host = mcp_host
        self.mcp_port = mcp_port
        
        self.mcp_connection = None
        self.executor = None
        
    async def start_enhanced_client(self):
        """Start client with enhanced debug capabilities"""
        print("🚀 Starting Enhanced Llammy MCP Client with Debug/Regen...")
        
        # Connect to MCP
        uri = f"ws://{self.mcp_host}:{self.mcp_port}"
        self.mcp_connection = await websockets.connect(uri)
        
        # Create enhanced executor
        self.executor = EnhancedMCPExecutor(self, self.mcp_connection)
        
        print("✅ Enhanced client ready with human-in-the-loop debugging")
        
        # Start enhanced decision loop
        await self.enhanced_decision_loop()
    
    async def enhanced_decision_loop(self):
        """Enhanced decision loop with recovery capabilities"""
        while True:
            try:
                user_input = input("\n💭 What should I do? (or 'stats' for statistics): ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    stats = self.executor.get_execution_stats()
                    print(f"\n📊 Execution Statistics:")
                    print(f"   Total Actions: {stats['total']}")
                    print(f"   Success Rate: {stats['success_rate']:.1%}")
                    print(f"   Human Reviews: {stats['human_reviews']}")
                    print(f"   Avg Execution Time: {stats['avg_execution_time']:.2f}s")
                    continue
                
                # Get AI decision (simplified for example)
                decision = await self.ask_llammy(user_input)
                actions = decision.get('actions', [])
                
                if actions:
                    print(f"\n🎬 Executing {len(actions)} actions with recovery...")
                    
                    for action in actions:
                        result = await self.executor.execute_action_with_recovery(
                            action, 
                            {}  # Current scene state would go here
                        )
                        
                        if result.status == ActionStatus.SUCCESS:
                            print(f"✅ Action succeeded: {action.get('method')}")
                        elif result.status == ActionStatus.FAILED:
                            print(f"❌ Action failed after recovery: {result.error}")
                        elif result.status == ActionStatus.HUMAN_REVIEW:
                            print(f"👤 Action requires manual intervention: {action.get('method')}")
                
            except KeyboardInterrupt:
                print("\n👋 Shutting down enhanced client...")
                break
            except Exception as e:
                print(f"❌ Client error: {e}")
    
    async def ask_llammy(self, prompt: str) -> Dict[str, Any]:
        """Simplified Llammy interface for example"""
        # This would use your existing Ollama integration
        return {
            'actions': [
                {
                    'method': 'create_object',
                    'params': {'name': 'TestCube', 'type': 'MESH'},
                    'expected_outcome': 'Create a test cube'
                }
            ]
        }

# Export classes for integration
__all__ = [
    'EnhancedMCPExecutor',
    'EmbeddedDebugger', 
    'HumanReviewInterface',
    'MCPActionValidator',
    'FailureContext',
    'ActionResult',
    'FailureType',
    'ActionStatus'
]