class Router:

    def __init__(self):

        self.stack = None

    def set_stack(self, stack):

        self.stack = stack

    def goto(self, index):

        if self.stack:

            self.stack.setCurrentIndex(index)