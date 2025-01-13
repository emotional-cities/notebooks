import re
from unidecode import unidecode


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

def extract_session_name(folder_name):
    """Fetch corret session name from the original folder name
    Extract the first string between underscores in the folder name.
    Normalize special characters like 'ç' to 'c'.
    
    Args:
        folder_name (str): Folder name to extract the session name.
    
    Returns:
        str: Normalized session name or the original folder name if no match is found.
    """
    match = re.search(r'_(.*?)_', folder_name)  # Find first string between underscores
    if match:
        session_name = match.group(1)
        return unidecode(session_name)  # Normalize special characters
    return unidecode(folder_name)  # Fallback to the original folder name