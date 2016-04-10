def call_simulator(simu, action):
    output = simu(action)
    if hasattr(output, '__len__') and len(output) == 2:
        t = output[0]
        x = output[1]
    else:
        t = output
        x = None  # self.test.X[action, :]
    return t, x
