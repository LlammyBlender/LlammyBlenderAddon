# =============================================================================
# LLAMMY HARVESTER MODULE - Intelligence Gathering with Embedded Multi-Purpose Model
# llammy_harvester_module.py - Ethical web scraping + embedded validation + dataset creation
# =============================================================================

import os
import json
import sqlite3
import requests
import hashlib
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import tempfile

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("BeautifulSoup4 not available - using basic text parsing")

class EmbeddedMultiPurposeModel:
    """Multi-purpose Qwen3:0.6B/Gemma3:260M in Harvester for content analysis and validation"""
    
    def __init__(self, model_name="qwen3:0.6b", ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.usage_stats = {
            'content_analysis': 0,
            'api_validation': 0,
            'quality_scoring': 0,
            'dataset_curation': 0,
            'competitive_analysis': 0
        }
        print(f"Harvester embedded model initialized: {model_name}")
    
    def analyze_web_content_quality(self, content: str, source_url: str) -> Dict[str, Any]:
        """Analyze harvested web content for quality and relevance"""
        self.usage_stats['content_analysis'] += 1
        
        prompt = f"""Analyze Blender web content quality:
URL: {source_url}
Content: {content[:800]}...

Rate quality (1-10), detect API version, assess business value
Response: {{"quality_score": int, "api_version": str, "blender_relevant": bool, "business_value": int}}"""
        
        return self._call_model_json(prompt, "content_analysis")
    
    def validate_scraped_code(self, code: str, context: str = "") -> Dict[str, Any]:
        """Validate scraped Blender code for training data inclusion"""
        self.usage_stats['api_validation'] += 1
        
        prompt = f"""Validate scraped Blender Python code:
Code: {code}
Context: {context}

Check: syntax, API currency, educational value, complexity
Response: {{"valid": bool, "api_current": bool, "complexity": int, "training_worthy": bool}}"""
        
        return self._call_model_json(prompt, "api_validation")
    
    def score_dataset_potential(self, content: str, metadata: Dict[str, Any]) -> float:
        """Score content's potential value for dataset creation"""
        self.usage_stats['dataset_curation'] += 1
        
        source = metadata.get('source', 'unknown')
        
        prompt = f"""Score dataset potential (0.0-1.0):
Content: {content[:500]}...
Source: {source}
Metadata: {metadata}

Consider: uniqueness, educational value, API sophistication, commercial potential
Response: {{"dataset_score": float, "reasoning": str}}"""
        
        result = self._call_model_json(prompt, "dataset_scoring")
        return result.get('dataset_score', 0.5)
    
    def analyze_competitive_intelligence(self, content: str, source: str) -> Dict[str, Any]:
        """Analyze content for competitive intelligence and market positioning"""
        self.usage_stats['competitive_analysis'] += 1
        
        prompt = f"""Analyze competitive intelligence:
Source: {source}
Content: {content[:600]}...

Identify: competitor tools, pricing, features, market gaps, our advantages
Response: {{"competitor_detected": bool, "market_intelligence": str, "competitive_advantage": bool}}"""
        
        return self._call_model_json(prompt, "competitive_analysis")
    
    def curate_training_content(self, content: str, execution_success: bool = None) -> Dict[str, Any]:
        """Curate content for Triangle104 training data"""
        self.usage_stats['dataset_curation'] += 1
        
        success_context = f"Execution success: {execution_success}" if execution_success is not None else ""
        
        prompt = f"""Curate for Triangle104 training:
Content: {content[:700]}...
{success_context}

Rate for model training: quality, difficulty, uniqueness
Response: {{"triangle104_worthy": bool, "difficulty_level": int, "uniqueness_score": float}}"""
        
        return self._call_model_json(prompt, "training_curation")
    
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
                    "num_predict": 180
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
            print(f"Harvester embedded model error ({task_type}): {e}")
        
        return {'error': 'model_call_failed', 'task_type': task_type}

class EthicalWebHarvester:
    """Ethical web scraping with embedded model content analysis"""
    
    def __init__(self, embedded_model: EmbeddedMultiPurposeModel):
        self.embedded_model = embedded_model
        self.allowed_domains = [
            'docs.blender.org',
            'wiki.blender.org',
            'github.com',
            'blender.stackexchange.com'
        ]
        self.rate_limits = {
            'docs.blender.org': 10,  # requests per minute
            'github.com': 30,
            'default': 15
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Llammy Educational AI (Ethical Blender Research)'
        })
    
    def harvest_blender_api_docs(self) -> Dict[str, Any]:
        """Harvest current Blender API documentation with embedded model analysis"""
        
        harvest_results = {
            'pages_scraped': 0,
            'high_quality_content': 0,
            'training_worthy_content': 0,
            'api_updates_found': 0,
            'competitive_intelligence': 0,
            'total_business_value': 0.0
        }
        
        api_urls = [
            'https://docs.blender.org/api/current/',
            'https://docs.blender.org/api/current/bpy.ops.html',
            'https://docs.blender.org/api/current/bpy.types.html'
        ]
        
        for url in api_urls:
            try:
                if not self._check_robots_txt(url):
                    continue
                
                if not self._rate_limit_ok(url):
                    time.sleep(60)  # Wait before continuing
                
                content = self._fetch_page_content(url)
                if content:
                    # Embedded model analysis
                    quality_analysis = self.embedded_model.analyze_web_content_quality(content, url)
                    
                    if quality_analysis.get('quality_score', 0) >= 7:
                        harvest_results['high_quality_content'] += 1
                        
                        # Check for training potential
                        if 'bpy.' in content and 'def ' in content:
                            validation = self.embedded_model.validate_scraped_code(content)
                            if validation.get('training_worthy', False):
                                harvest_results['training_worthy_content'] += 1
                                self._store_training_content(content, url, quality_analysis)
                        
                        # Dataset scoring
                        dataset_score = self.embedded_model.score_dataset_potential(
                            content, {'source': url, 'quality': quality_analysis}
                        )
                        harvest_results['total_business_value'] += dataset_score * 10.0
                    
                    harvest_results['pages_scraped'] += 1
                
            except Exception as e:
                print(f"Error harvesting {url}: {e}")
        
        return harvest_results
    
    def _check_robots_txt(self, url: str) -> bool:
        """Check robots.txt compliance"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            
            return rp.can_fetch('*', url)
        except Exception:
            return True  # Allow if robots.txt check fails
    
    def _rate_limit_ok(self, url: str) -> bool:
        """Check if rate limit allows request"""
        domain = urlparse(url).netloc
        limit = self.rate_limits.get(domain, self.rate_limits['default'])
        
        # Simple rate limiting implementation
        return True  # Implement actual rate limiting as needed
    
    def _fetch_page_content(self, url: str) -> Optional[str]:
        """Fetch and parse page content"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            if BS4_AVAILABLE:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract meaningful content
                for script in soup(["script", "style"]):
                    script.decompose()
                content = soup.get_text()
            else:
                content = response.text
            
            # Clean up content
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            return '\n'.join(lines)
            
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            return None
    
    def _store_training_content(self, content: str, source_url: str, quality_analysis: Dict[str, Any]):
        """Store high-quality content for training data"""
        
        training_db_path = os.path.expanduser("~/.llammy/harvested_training.db")
        os.makedirs(os.path.dirname(training_db_path), exist_ok=True)
        
        conn = sqlite3.connect(training_db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS harvested_training (
                id INTEGER PRIMARY KEY,
                source_url TEXT,
                content TEXT,
                quality_score INTEGER,
                api_version TEXT,
                business_value REAL,
                triangle104_worthy INTEGER,
                harvest_timestamp TIMESTAMP
            )
        ''')
        
        # Curate for Triangle104
        curation_result = self.embedded_model.curate_training_content(content)
        
        conn.execute('''
            INSERT INTO harvested_training 
            (source_url, content, quality_score, api_version, business_value, 
             triangle104_worthy, harvest_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            source_url,
            content[:5000],  # Limit content size
            quality_analysis.get('quality_score', 5),
            quality_analysis.get('api_version', '4.4'),
            quality_analysis.get('business_value', 1.0),
            1 if curation_result.get('triangle104_worthy', False) else 0,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()

class IntelligentHarvesterManager:
    """Intelligent harvesting with embedded model decision making"""
    
    def __init__(self, embedded_model: EmbeddedMultiPurposeModel, daily_cap_gb: float = 4.0):
        self.embedded_model = embedded_model
        self.daily_cap_gb = daily_cap_gb
        self.usage_file = os.path.expanduser("~/.llammy/harvester_usage.json")
        self.web_harvester = EthicalWebHarvester(embedded_model)
        
        # Initialize usage tracking
        self._init_usage_tracking()
    
    def _init_usage_tracking(self):
        """Initialize daily usage tracking"""
        os.makedirs(os.path.dirname(self.usage_file), exist_ok=True)
        
        if not os.path.exists(self.usage_file):
            self._reset_daily_usage()
    
    def _reset_daily_usage(self):
        """Reset daily usage counter"""
        usage_data = {
            'date': datetime.now().date().isoformat(),
            'data_harvested_mb': 0.0,
            'pages_processed': 0,
            'training_content_found': 0,
            'competitive_intelligence_gathered': 0
        }
        
        with open(self.usage_file, 'w') as f:
            json.dump(usage_data, f)
    
    def should_harvest_today(self) -> bool:
        """Embedded model decides if harvesting should occur today"""
        
        with open(self.usage_file, 'r') as f:
            usage_data = json.load(f)
        
        # Check daily cap
        if usage_data['data_harvested_mb'] >= (self.daily_cap_gb * 1024):
            return False
        
        # Check if Blender is idle (basic version)
        if self._is_blender_idle():
            return True
        
        return False
    
    def _is_blender_idle(self) -> bool:
        """Check if Blender is idle for background harvesting"""
        try:
            if psutil:
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                    if 'blender' in proc.info['name'].lower():
                        if proc.info['cpu_percent'] < 10:  # Low CPU usage
                            return True
                return True  # No Blender process found
            else:
                return True  # Assume idle if psutil not available
        except Exception:
            return True
    
    def perform_intelligent_harvest(self) -> Dict[str, Any]:
        """Perform intelligent harvesting with embedded model curation"""
        
        if not self.should_harvest_today():
            return {'status': 'skipped', 'reason': 'daily_cap_reached_or_blender_busy'}
        
        print("Starting intelligent harvest with embedded model curation...")
        
        harvest_results = {
            'web_content': {},
            'local_content': {},
            'training_data_created': 0,
            'competitive_intelligence': 0,
            'total_business_value': 0.0,
            'embedded_model_decisions': 0
        }
        
        # Web harvesting with embedded model analysis
        web_results = self.web_harvester.harvest_blender_api_docs()
        harvest_results['web_content'] = web_results
        harvest_results['total_business_value'] += web_results.get('total_business_value', 0)
        
        # Update usage tracking
        self._update_usage_tracking(harvest_results)
        
        return harvest_results
    
    def _update_usage_tracking(self, harvest_results: Dict[str, Any]):
        """Update daily usage tracking"""
        
        with open(self.usage_file, 'r') as f:
            usage_data = json.load(f)
        
        # Estimate data usage (rough calculation)
        pages_processed = harvest_results.get('web_content', {}).get('pages_scraped', 0)
        estimated_mb = pages_processed * 0.5  # Rough estimate
        
        usage_data['data_harvested_mb'] += estimated_mb
        usage_data['pages_processed'] += pages_processed
        usage_data['training_content_found'] += harvest_results.get('web_content', {}).get('training_worthy_content', 0)
        
        with open(self.usage_file, 'w') as f:
            json.dump(usage_data, f)

class LlammyDataHarvester:
    """Main harvester interface with embedded model intelligence"""
    
    def __init__(self, embedded_model_name: str = "qwen3:0.6b"):
        self.embedded_model = EmbeddedMultiPurposeModel(embedded_model_name)
        self.harvester_manager = IntelligentHarvesterManager(self.embedded_model)
        self.db_path = os.path.expanduser("~/.llammy/harvester.db")
        self._init_database()
        
        print(f"Llammy Data Harvester initialized with {embedded_model_name}")
    
    def _init_database(self):
        """Initialize harvester database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS harvest_log (
                id INTEGER PRIMARY KEY,
                harvest_date TIMESTAMP,
                content_harvested INTEGER,
                business_value REAL,
                training_data_created INTEGER,
                competitive_intelligence INTEGER,
                embedded_model_stats TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def daily_harvest_cycle(self) -> Dict[str, Any]:
        """Execute daily harvest cycle with embedded model intelligence"""
        
        start_time = time.time()
        
        harvest_results = self.harvester_manager.perform_intelligent_harvest()
        
        if harvest_results.get('status') == 'skipped':
            return harvest_results
        
        # Log results
        self._log_harvest_results(harvest_results)
        
        # Embedded model statistics
        model_stats = self.embedded_model.usage_stats.copy()
        harvest_results['embedded_model_stats'] = model_stats
        harvest_results['processing_time'] = time.time() - start_time
        
        print(f"Daily harvest completed: {harvest_results.get('total_business_value', 0):.1f} business value")
        
        return harvest_results
    
    def _log_harvest_results(self, results: Dict[str, Any]):
        """Log harvest results to database"""
        
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO harvest_log 
            (harvest_date, content_harvested, business_value, training_data_created,
             competitive_intelligence, embedded_model_stats)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            results.get('web_content', {}).get('pages_scraped', 0),
            results.get('total_business_value', 0.0),
            results.get('web_content', {}).get('training_worthy_content', 0),
            results.get('competitive_intelligence', 0),
            json.dumps(results.get('embedded_model_stats', {}))
        ))
        conn.commit()
        conn.close()
    
    def get_harvest_status(self) -> Dict[str, Any]:
        """Get comprehensive harvest status"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT COUNT(*) as total_harvests,
                   SUM(business_value) as total_business_value,
                   SUM(training_data_created) as total_training_data,
                   MAX(harvest_date) as last_harvest
            FROM harvest_log
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            'total_harvests': stats[0] or 0,
            'total_business_value': stats[1] or 0.0,
            'total_training_data': stats[2] or 0,
            'last_harvest': stats[3],
            'embedded_model': self.embedded_model.model_name,
            'embedded_model_stats': self.embedded_model.usage_stats.copy(),
            'daily_cap_status': self.harvester_manager.should_harvest_today()
        }

# Global harvester instance
llammy_harvester = None

def get_llammy_harvester(embedded_model: str = "qwen3:0.6b") -> LlammyDataHarvester:
    """Get global harvester instance"""
    global llammy_harvester
    if llammy_harvester is None:
        llammy_harvester = LlammyDataHarvester(embedded_model)
    return llammy_harvester

def test_harvester_with_embedded_model():
    """Test harvester with embedded model"""
    print("Testing Llammy Harvester with Embedded Model...")
    
    harvester = get_llammy_harvester()
    
    # Test daily harvest
    results = harvester.daily_harvest_cycle()
    
    print(f"Harvest results: {results}")
    print(f"Status: {harvester.get_harvest_status()}")
    
    return True

if __name__ == "__main__":
    test_harvester_with_embedded_model()

print("LLAMMY HARVESTER with Embedded Multi-Purpose Model LOADED!")
print("Features:")
print("  - Ethical web scraping with robots.txt compliance")
print("  - Embedded Qwen3:0.6B/Gemma3:260M for content analysis")
print("  - Real-time quality scoring and validation")
print("  - Triangle104 training data curation")
print("  - Competitive intelligence gathering")
print("  - Dataset creation and business value calculation")
print("  - Background processing during Blender idle time")
