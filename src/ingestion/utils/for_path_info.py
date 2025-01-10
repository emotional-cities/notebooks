import re

def fetch_path_num(data_path):
    path = str(data_path)
    path = path.split('\\')
    filename = path[-1]
    match = re.search(r'1(\d{2})', filename)
    if match:
        # The group(1) method returns the matched string
        numbers = match.group(1)
        print(numbers)
    return numbers