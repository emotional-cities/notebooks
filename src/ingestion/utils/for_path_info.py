import re
from unidecode import unidecode

def fetch_path_num(data_path):
    """Fetch the session number from the data path.
    Extract the session number from the data path.
    It can be exctracted from original data_path or from the session name.
    """
    if '_' in data_path:
        path = str(data_path)
        path = path.split('\\')
        filename = path[-1]
        match = re.search(r'1(\d{2})', filename)
        if match:
            # The group(1) method returns the matched string
            numbers = match.group(1)
            print(numbers)
    else:        
        # Load session information
        sessions = [
            ('Baixa', 4),
            ('Belem', 1),
            ('Parque', 6),
            ('Gulbenkian', 3),
            ('Lapa', 2),
            ('Graca', 5),
            ('Gulb1', 7),
            ('Casamoeda', 8),
            ('Agudo', 9),
            ('Msoares', 10),
            ('Marvila', 11),
            ('Oriente', 12),
            ('Madre', 13),
            ('Pupilos', 14),
            ('Luz', 15),
            ('Alfarrobeira', 16),
            ('Restauradores', 17),
            ('Restelo', 18),
            ('Estrela', 19),
            ('EstrelaA', 20),
            ('EstrelaB', 21),
            ('Prazeres', 22)            
        ]
        # Get number from sessions
        for session_name, session_number in sessions:
            if session_name.lower() in data_path.lower():
                numbers = session_number
    
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

