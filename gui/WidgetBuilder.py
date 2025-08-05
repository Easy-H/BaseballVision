import sys
import tkinter.scrolledtext

class WidgetBuilder:
    def set_window(self, target, data):
        if "title" in data:
            target.title(data["title"])
        if "geometry" in data:
            target.geometry(data["geometry"])
        if "resizable" in data:
            target.resizable(data["resizable"], data["resizable"])
        if "background" in data:
            target.configure(background=data["background"])

    def create_widget(self, master, data):
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

    def widget_config(self, obj_name, key, value):
        if obj_name not in self.objs:
            return
        self.objs[obj_name][key] = value

    def widgets_config(self, obj_name_list, key, value):
        for obj_name in obj_name_list:
            self.widget_config(obj_name, key, value)