from datetime import datetime
import threading
import time
from typing import Optional, Dict
import logging
import psutil

class HeartbeatMonitor:
    def __init__(self, logger: logging.Logger, heartbeat_interval: int = 30, 
                 warning_threshold: int = 60,
                 critical_threshold: int = 90):
        self.logger = logger
        
        # Configuration
        self.heartbeat_interval = heartbeat_interval  # seconds between checks
        self.warning_threshold = warning_threshold    # seconds before warning
        self.critical_threshold = critical_threshold  # seconds before critical alert
        
        # State tracking
        self.last_heartbeat: Optional[datetime] = None
        self.is_running: bool = False
        self.system_status: str = "INITIALIZING"
        self.monitor_thread: Optional[threading.Thread] = None
        self.system_metrics: Dict = {}

    def start_monitoring(self):
        """Start independent monitoring thread"""
        self.logger.info("Starting heartbeat monitor")
        self.is_running = True
        self.system_status = "RUNNING"
        
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name='HeartbeatMonitor'
        )
        self.monitor_thread.start()
        
    def _monitor_loop(self):
        """Independent monitoring loop"""
        while self.is_running:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Verify system health
                self._verify_system_health()
                
                # Update heartbeat if system is healthy
                self._update_heartbeat()
                
                # Wait for next interval
                time.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Monitor loop error: {str(e)}")
                self.system_status = "ERROR"

    def _update_system_metrics(self):
        """Update current system metrics"""
        try:
            self.system_metrics = {
                'timestamp': datetime.now(),
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'system_status': self.system_status
            }
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {str(e)}")

    def _verify_system_health(self):
        """Verify system is healthy"""
        try:
            # Check system resources
            if self.system_metrics['cpu_usage'] > 80:
                self.logger.warning(f"High CPU usage: {self.system_metrics['cpu_usage']}%")
                self.system_status = "WARNING"
                
            if self.system_metrics['memory_usage'] > 80:
                self.logger.warning(f"High memory usage: {self.system_metrics['memory_usage']}%")
                self.system_status = "WARNING"
                
            # Check last heartbeat
            if self.last_heartbeat:
                time_since_last = (datetime.now() - self.last_heartbeat).total_seconds()
                
                if time_since_last > self.critical_threshold:
                    self.system_status = "CRITICAL"
                    self.trigger_emergency_protocol(time_since_last)
                elif time_since_last > self.warning_threshold:
                    self.system_status = "WARNING"
                    self.send_warning_alert(time_since_last)
                    
        except Exception as e:
            self.logger.error(f"Health verification failed: {str(e)}")
            self.system_status = "ERROR"

    def _update_heartbeat(self):
        """Update heartbeat timestamp"""
        self.last_heartbeat = datetime.now()
        self.logger.debug(f"Heartbeat updated: {self.last_heartbeat}")
        self.logger.debug(f"System status: {self.system_status}")

    def trigger_emergency_protocol(self, delay_seconds: float):
        """Handle critical system state"""
        try:
            self.logger.critical(
                f"CRITICAL: System non-responsive for {delay_seconds} seconds"
            )
            
            # Log detailed system state for diagnosis
            emergency_data = {
                'timestamp': datetime.now(),
                'delay_seconds': delay_seconds,
                'memory_usage': self.system_metrics['memory_usage'],
                'cpu_usage': self.system_metrics['cpu_usage'],
                'last_heartbeat': self.last_heartbeat
            }
            self.logger.critical(f"Emergency Protocol Triggered: {emergency_data}")
            
            # Write emergency state to disk
            self._write_emergency_state(emergency_data)
            
            # 4. Signal trading system to stop new entries
            # (Trading system checks status before entries)

            
        except Exception as e:
            self.logger.error(f"Emergency protocol failed: {str(e)}")

    def send_warning_alert(self, delay_seconds: float):
        """Handle warning system state"""
        self.logger.warning(
            f"WARNING: System delayed for {delay_seconds} seconds"
        )
        # TODO: Implement warning alerts
        # - Send warning notifications
        # - Log detailed system state

    def get_status(self) -> Dict:
        """Get current system status and metrics"""
        return {
            'status': self.system_status,
            'last_heartbeat': self.last_heartbeat,
            'metrics': self.system_metrics
        }

    def stop_monitoring(self):
        """Safely stop the monitor"""
        self.logger.info("Stopping heartbeat monitor")
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
