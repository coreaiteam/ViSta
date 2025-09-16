import psutil
import time
import logging
import platform
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AdvancedResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.system_info = self._get_system_info()
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.initial_memory: Optional[int] = None
        self.final_memory: Optional[int] = None
        self.initial_cpu: Optional[float] = None
        self.final_cpu: Optional[float] = None
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get detailed system hardware information"""
        try:
            # CPU information
            cpu_freq = psutil.cpu_freq()
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False) or 1,
                'total_cores': psutil.cpu_count(logical=True) or 1,
                'max_frequency': f"{cpu_freq.max:.2f} MHz" if cpu_freq else "N/A",
                'cpu_name': platform.processor(),
                'architecture': platform.machine()
            }
            
            # Memory information
            virtual_mem = psutil.virtual_memory()
            memory_info = {
                'total_ram': f"{virtual_mem.total / (1024**3):.2f} GB",
                'available_ram': f"{virtual_mem.available / (1024**3):.2f} GB",
                'ram_type': self._get_ram_type()
            }
            
            # System information
            system_info = {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'platform': platform.platform()
            }
            
            return {
                'cpu': cpu_info,
                'memory': memory_info,
                'system': system_info
            }
            
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {
                'cpu': {'physical_cores': 1, 'total_cores': 1},
                'memory': {'total_ram': 'N/A', 'available_ram': 'N/A'},
                'system': {'system': 'Unknown', 'release': 'Unknown'}
            }
    
    def _get_ram_type(self) -> str:
        """Try to get RAM type"""
        try:
            if platform.system() == "Linux":
                return "DDR4"  # مقدار پیش‌فرض
            elif platform.system() == "Windows":
                return "DDR4"  # مقدار پیش‌فرض
            else:
                return "Unknown"
        except:
            return "Unknown"
    
    def __enter__(self):
        # اطلاعات قبل از اجرا
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info().rss
        self.initial_cpu = self.process.cpu_percent(interval=None)
        
        # اطلاعات سیستم در شروع
        self.initial_system_memory = psutil.virtual_memory().percent
        self.initial_system_cpu = psutil.cpu_percent(interval=None)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # همیشه end_time را تنظیم کن حتی اگر exception اتفاق افتاده باشد
        self.end_time = time.time()
        
        try:
            # اطلاعات بعد از اجرا
            self.final_memory = self.process.memory_info().rss
            self.final_cpu = self.process.cpu_percent(interval=None)
            
            # اطلاعات سیستم در پایان
            self.final_system_memory = psutil.virtual_memory().percent
            self.final_system_cpu = psutil.cpu_percent(interval=0.1)
            
            # گزارش کامل
            self._log_comprehensive_report()
            
        except Exception as e:
            # اگر خطایی در مانیتورینگ اتفاق افتاد، فقط log کن و ادامه بده
            logger.error(f"Error in resource monitoring: {e}")
    
    def _log_comprehensive_report(self):
        """Log a comprehensive resource usage report"""
        if self.start_time is None or self.end_time is None:
            logger.error("Resource monitoring times are not set")
            return
        
        duration = self.end_time - self.start_time
        
        # محاسبات حافظه با بررسی null
        memory_usage_mb = self.final_memory / 1024 / 1024 if self.final_memory else 0
        memory_delta_mb = (self.final_memory - self.initial_memory) / 1024 / 1024 if self.final_memory and self.initial_memory else 0
        
        # محاسبه CPU usage
        cpu_cores = self.system_info['cpu']['total_cores']
        normalized_cpu_percent = ((self.final_cpu or 0) - (self.initial_cpu or 0)) / cpu_cores if cpu_cores else 0

        report = f"""
{'='*60}
CLUSTERING ENGINE - RESOURCE USAGE REPORT
{'='*60}

SYSTEM SPECIFICATIONS:
• CPU: {self.system_info['cpu'].get('cpu_name', 'Unknown')}
• Cores: {self.system_info['cpu'].get('physical_cores', 1)} physical, {self.system_info['cpu'].get('total_cores', 1)} logical
• Max Frequency: {self.system_info['cpu'].get('max_frequency', 'N/A')}
• Architecture: {self.system_info['cpu'].get('architecture', 'Unknown')}
• Total RAM: {self.system_info['memory'].get('total_ram', 'N/A')}
• Available RAM: {self.system_info['memory'].get('available_ram', 'N/A')}
• System: {self.system_info['system'].get('system', 'Unknown')} {self.system_info['system'].get('release', 'Unknown')}

PERFORMANCE METRICS:
• Execution Time: {duration:.4f} seconds
• Process CPU Usage: {normalized_cpu_percent:.2f}% (normalized per core)
• Process Memory Usage: {memory_usage_mb:.2f} MB
• Memory Delta: {memory_delta_mb:+.2f} MB
• Peak Process Memory: {self.process.memory_info().rss / 1024 / 1024:.2f} MB

SYSTEM-WIDE USAGE:
• System CPU Usage: {(self.final_system_cpu or 0):.2f}%
• System Memory Usage: {(self.final_system_memory or 0):.2f}%

PROCESS DETAILS:
• Process ID: {self.process.pid}
• Process Name: {self.process.name()}
• Process Status: {self.process.status()}
{'='*60}
"""
        
        logger.info(report)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics as a dictionary for programmatic access"""
        if self.start_time is None or self.end_time is None:
            return {'error': 'Monitoring not completed'}
        
        duration = self.end_time - self.start_time
        memory_delta_mb = (self.final_memory - self.initial_memory) / 1024 / 1024 if self.final_memory and self.initial_memory else 0
        
        return {
            'execution_time': duration,
            'process_cpu_usage': (self.final_cpu or 0) - (self.initial_cpu or 0),
            'process_memory_usage_mb': (self.final_memory or 0) / 1024 / 1024,
            'memory_delta_mb': memory_delta_mb,
            'system_cpu_usage': self.final_system_cpu or 0,
            'system_memory_usage': self.final_system_memory or 0,
            'system_info': self.system_info
        }