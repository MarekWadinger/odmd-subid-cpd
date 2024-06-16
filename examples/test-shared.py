import threading

class SharedVariableCounter:
    def __init__(self):
        self.shared_variable = 0
        self.lock = threading.Lock()

    def increment_shared_variable(self):
        with self.lock:
            self.shared_variable += 1
            print("Shared variable:", self.shared_variable)

# Create an instance of the class
counter = SharedVariableCounter()

# Create multiple threads to increment the shared variable
threads = []
for _ in range(5):
    t = threading.Thread(target=counter.increment_shared_variable)
    t.start()
    threads.append(t)

# Wait for all threads to finish
for t in threads:
    t.join()
