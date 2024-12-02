import matplotlib.pyplot as plt
import random
import time

class VisualObject:
    def __init__(self, name, value, x, y, ax):
        self.name = name
        self.value = value
        self.x = x
        self.y = y
        self.text = ax.text(x, y, name, fontsize=value, ha='center', va='center')

    def update_value(self, value):
        self.value = value
        self.text.set_fontsize(value)

    def remove(self):
        self.text.remove()


fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

object_list = []

def add_object(name, value):
    existing_object = next((obj for obj in object_list if obj.name == name), None)
    if existing_object:
        existing_object.update_value(value)
    else:
        x, y = random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)
        new_object = VisualObject(name, value, x, y, ax)
        object_list.append(new_object)

def remove_object(name):
    global object_list
    for obj in object_list:
        if obj.name == name:
            obj.remove()
            object_list.remove(obj)
            break

def draw():
    global object_list
    plt.pause(0.01)

def show():
    plt.ion()

    for i in range(10):
        if random.random() > 0.3:
            obj_name = f"Obj{random.randint(1, 5)}"
            obj_value = random.randint(10, 40)
            add_object(obj_name, obj_value)
        else:
            if object_list:
                obj_to_remove = random.choice(object_list).name
                remove_object(obj_to_remove)
        draw()
        time.sleep(0.5)
    print(2)
    plt.ioff()
    plt.show()

show()






