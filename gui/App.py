import tkinter
import sys

class App(tkinter.Tk):
    def __init__(self, master):
        pass
    def create_widget(self, master, data):
        if data is None:
            return
        master.title(data["title"])
        master.geometry(data["geometry"])
        master.resizable(data["resizable"], data["resizable"])
        self.objs = {}
        self._create_widget(master, data)

    def _create_widget(self, parent, data):
        if "objs" not in data:
            return
        data = data["objs"]
        for j in data:
            self._widget(parent, j)
    
    def _widget(self, parent, data):
        class_ = getattr(sys.modules[data["module"]], data["type"])
        obj = class_(parent)

        self._widget_attr(obj, data)
        self._widget_method(obj, data)

        self.objs[data["name"]] = obj
        self._create_widget(obj, data)

    def _widget_attr(self, obj, data):
        if "attr" not in data:
            return
        for label, attr in data["attr"].items():
            obj[label] = attr

    def _widget_method(self, obj, data):
        if "method" not in data:
            return
        for method in data["method"]:
            getattr(obj, method["name"])(**method["attr"])

    def widget_config(self, name, key, value):
        if name not in self.objs:
            return
        self.objs[name][key] = value

    def set_widget_state(self, obj_list, state):
        for obj in obj_list:
            self.widget_config(obj, "state", state)