class materials:

    def __init__(self, mat_name):
        self.name = mat_name

    def write(self):
        return self.name
    
    def basic_structure(structure):
        return structure


new = materials("Aluminium")

print(new.basic_structure())
