import tkinter
import sys

class AppTopLevel(tkinter.Toplevel):
    def __init__(self, master):
        super().__init__(master)
    def create_widget(self, master, data):
        if data is None:
            return
        self.title(data["title"])
        self.geometry(data["geometry"])
        self.resizable(data["resizable"], data["resizable"])
        
        self.objs = {}
        self._create_widget(master, data)

    def _create_widget(self, parent, data):
        if "objs" not in data:
            return
        data = data["objs"]
        for j in data:
            self._widget(parent, j)
    
    def _widget(self, parent, data):
        if "module" not in data:
            return
        if "type" not in data:
            return
        
        class_ = getattr(sys.modules[data["module"]], data["type"])
        obj = class_(parent)

        self._widget_attr(obj, data)
        self._widget_method(obj, data)
        self._widget_add(obj, data)

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

    def _widget_add(self, obj, data):
        if "name" not in data:
            return
        self.objs[data["name"]] = obj

    def set_widget_state(self, obj_list, state):
        for obj in obj_list:
            obj.config(state=state)