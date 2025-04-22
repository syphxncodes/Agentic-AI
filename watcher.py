import os
import time
import hashlib
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Files to ignore (output files written by main.py)
IGNORE_FILES = {"medicines.csv", "equipment.csv", "updated_medical_supply.csv"}
COOLDOWN_SECONDS = 5  # Minimum time between triggers

class ContentChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.file_hashes = {}
        self.last_trigger = 0

    def get_file_hash(self, path):
        try:
            with open(path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None

    def on_modified(self, event):
        self._handle_event(event)

    def _handle_event(self, event):
        if event.is_directory:
            return

        filename = os.path.basename(event.src_path)
        if filename in IGNORE_FILES:
            return

        # Only process CSV files
        if not filename.lower().endswith('.csv'):
            return

        # Check cooldown
        if time.time() - self.last_trigger < COOLDOWN_SECONDS:
            return

        # Get current and previous hashes
        current_hash = self.get_file_hash(event.src_path)
        previous_hash = self.file_hashes.get(event.src_path)

        # Only trigger if content changed
        if current_hash and current_hash != previous_hash:
            print(f"[Real content change detected] {filename}")
            try:
                subprocess.run(["python", "main.py"])
            except Exception as e:
                print(f"Error running main.py: {e}")

            # Update hash and timestamp
            self.file_hashes[event.src_path] = current_hash
            self.last_trigger = time.time()

if __name__ == "__main__":
    path_to_watch = r"D:\Agentic AI Hackathon"  # ← Raw string for Windows paths
    if not os.path.isdir(path_to_watch):
        print(f"Folder '{path_to_watch}' does not exist.")
        exit(1)

    event_handler = ContentChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=path_to_watch, recursive=True)
    observer.start()

    print(f"🔍 Watching folder: {path_to_watch} (recursively)")
    print(f"🚫 Ignoring files: {', '.join(IGNORE_FILES)}")
    print("⚙️  Triggering only on actual content changes (ignoring metadata)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping watcher...")
        observer.stop()
    observer.join()
