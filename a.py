with open('requirements.txt', 'r') as fh:
    requirements = [line.strip() for line in fh.readlines()]
print(requirements)
