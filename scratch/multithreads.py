import weakref
import threading

class A:
    pass

def finalizer(i):
    print("in finalizer: ", i)

def make_a(i):
    a = A()
    weakref.finalize(a, finalizer, i)

tids = [threading.Thread(target=make_a, args=(i,)) for i in range(10)]
for t in tids:
    t.start()
for t in tids:
    t.join()

