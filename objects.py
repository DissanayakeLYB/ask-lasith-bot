class AgentRun:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def cal(num1, num2):
        return (num1 + num2)
    
    def power(self, num):
        return num**self.age

class AgentFinish:

    func = "ended"

new = AgentRun("Lasith", 25)

print(new.power(5))
