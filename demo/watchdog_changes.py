import os
import time
import watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# set up watchdog to monitor changes to your app.py file and restart the Gradio server automatically

def on_any_event(event):
    if event.is_directory:
        return
    elif event.event_type == 'modified' and event.src_path.endswith('.py'):
        print(f'Restarting Gradio server...')
        os.system('pkill -f "python app.py"')
        time.sleep(1)
        os.system('python app.py &')

if __name__ == '__main__':
    event_handler = FileSystemEventHandler()
    event_handler.on_any_event = on_any_event
    observer = Observer()
    observer.schedule(event_handler, '.', recursive=True)
    observer.start()
    print(f'Watchdog is monitoring changes to app.py...')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
