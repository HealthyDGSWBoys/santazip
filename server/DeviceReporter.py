from threading import Thread
import subprocess
import threading
import os

class DeviceReporter:
    _singleton = None
    _currentVal = {
        'cpu': [0, 100.0],
        'gpu': [0, 100.0],
        'ram': [0, 100.0],
        # 'vram': [1, 10],
        # 'cpu_temp': [1, 10],
        # 'gpu_temp': [1, 10]
    }
    _onUpdate = None
    @staticmethod
    def getDeviceReport(self):
        return self._currentVal
    
    def __init__(self) -> None:
        if DeviceReporter._singleton == None:
            threading.Timer(1, self.update).start()
            DeviceReporter._singleton = self

    def update(self):
        if self._onUpdate is not None:
            cpu = float(os.popen("top -bn1 | grep \"Cpu(s)\" | sed \"s/.*, *\([0-9.]*\)%* id.*/\\1/\" | awk '{print 100 - $1}'").read())
            ram = float(os.popen("free | grep Mem | awk '{print $3/$2 * 100.0}'").read())
            gpu = float(os.popen("gpustat -cp").read().split('|')[1].split("%")[0][-5:])

            self._currentVal["cpu"][0] = cpu
            self._currentVal["ram"][0] = ram
            self._currentVal["gpu"][0] = gpu

            self._onUpdate(self._currentVal)
        threading.Timer(2, self.update).start()

    def setOnUpdate(self, lam):
        self._onUpdate = lam

reporter = DeviceReporter()

# top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}'

