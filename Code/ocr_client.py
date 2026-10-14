"""Persistent PaddleOCR worker for the live OpenVINO scanner.
Paddle is loaded once, not once per camera scan. OCR remains in ocr_env.
"""
import os
import sys
import json
import queue
import subprocess
import threading
import time

class PaddleOCRClient:
    def __init__(self, python_exe=r"ocr_env\Scripts\python.exe", worker_script="ocr_engine.py"):
        self.python_exe = os.path.abspath(python_exe)
        self.worker_script = os.path.abspath(worker_script)
        self.proc = None
        self.responses = queue.Queue()
        self.lock = threading.Lock()
        self.startup_error = None

    def _read_stdout(self, process):
        try:
            for line in process.stdout:
                line = line.strip()
                if not line.startswith('{'):
                    continue  # Paddle model logs on stdout are not protocol messages.
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(message, dict) and (message.get('status') == 'ready' or 'items' in message):
                    self.responses.put(message)
        except Exception as exc:
            self.responses.put({'status': 'error', 'message': str(exc), 'items': []})

    def _stop(self):
        proc = self.proc
        self.proc = None
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
            try:
                proc.stdout.close()
            except Exception:
                pass

    def _start(self, timeout=90):
        self._stop()
        self.responses = queue.Queue()
        if not os.path.isfile(self.python_exe) or not os.path.isfile(self.worker_script):
            return {'status':'error', 'message':'Missing ocr_env or ocr_engine.py', 'items':[]}
        try:
            creationflags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            self.proc = subprocess.Popen([self.python_exe, '-u', self.worker_script, '--worker'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                encoding='utf-8', errors='replace', bufsize=1, creationflags=creationflags)
            threading.Thread(target=self._read_stdout, args=(self.proc,), daemon=True).start()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    return {'status':'error', 'message':f'OCR worker exited (code {self.proc.returncode})', 'items':[]}
                try:
                    result = self.responses.get(timeout=min(0.5, max(0.01, deadline-time.monotonic())))
                except queue.Empty:
                    continue
                if result.get('status') == 'ready':
                    print('[OCR] Persistent PaddleOCR worker READY')
                    return None
                if result.get('status') == 'error':
                    return result
            self._stop()
            return {'status':'error', 'message':'OCR model startup timed out', 'items':[]}
        except Exception as exc:
            return {'status':'error', 'message':f'OCR startup: {exc}', 'items':[]}

    def recognize(self, image_path, timeout=55):
        if not os.path.isfile(image_path):
            return {'status':'error', 'message':f'Image not found: {image_path}', 'items':[]}
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                error = self._start()
                if error is not None:
                    return error
            try:
                self.proc.stdin.write(os.path.abspath(image_path) + '\n')
                self.proc.stdin.flush()
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if self.proc.poll() is not None:
                        raise RuntimeError('OCR worker stopped unexpectedly')
                    try:
                        response = self.responses.get(timeout=min(0.5, max(0.01, deadline-time.monotonic())))
                    except queue.Empty:
                        continue
                    if 'items' in response:
                        return response
                self._stop() # Prevent stale responses from being assigned to next frame.
                return {'status':'error', 'message':f'OCR processing timed out ({timeout}s)', 'items':[]}
            except Exception as exc:
                self._stop()
                return {'status':'error', 'message':str(exc), 'items':[]}

    def close(self):
        with self.lock:
            self._stop()
