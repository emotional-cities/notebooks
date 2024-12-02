def infer_participant_code(city, subject_id, session, stimulus):
    """
    Infers the participant code based on the city, subject ID, session, and stimulus.

    Parameters:
    - city (str): The city name ('lisbon' or 'copenhagen').
    - subject_id (int): The subject's ID number.
    - session (str): The session name.
    - stimulus (str): The stimulus identifier (can be an empty string if not applicable).

    Returns:
    - str: The inferred participant code.

    Raises:
    - ValueError: If the session name or stimulus is invalid.

    Example:
    --------
    # Example with stimulus
    participant_id = infer_participant_code('lisbon', 3, 'Baixa', 'D12')
    print(participant_id)  # Output: 'OE104003_1'

    # Example without stimulus
    participant_id = infer_participant_code('lisbon', 5, 'Parque', '')
    print(participant_id)  # Output: 'OE106005'
    """

    if city.lower() == 'lisbon':
        # Load session information
        sessions = [
            ('Baixa', 4),
            ('Belem', 1),
            ('Parque', 6),
            ('Gulbenkian', 3),
            ('Lapa', 2),
            ('Graca', 5)
        ]

        # Define checkpoints corresponding to session numbers
        checkpoints = [
            ['A7', 'C9', 'C10'],            # Belem - 1
            ['A16', 'C17', 'C18'],          # Lapa - 2
            ['A3', 'B4', 'B6', 'B7'],       # Gulbenkian - 3
            ['D12', 'D27', 'D26', 'B10'],   # Baixa - 4
            ['C31', 'C32', 'D31', 'D32'],   # Graca - 5
            ['A24', 'A25']                  # Parque das Nações - 6
        ]

        # Find the session number based on the session name
        ses_num = None
        for name, number in sessions:
            if name.lower() == session.lower():
                ses_num = number
                break
        if ses_num is None:
            raise ValueError('Invalid session name.')

        # Format the session number with leading zero
        path_num = f"{ses_num:02d}"

        # Format the subject ID with leading zeros (assuming 3-digit IDs)
        subject_id_str = f"{subject_id:03d}"

        # Get the stimulus index if a stimulus is provided
        chk_num = None
        if stimulus:
            # Get checkpoints for the session
            checkpoints_for_session = checkpoints[ses_num - 1]
            # Find the index of the stimulus
            for idx, stim in enumerate(checkpoints_for_session):
                if stim.lower() == stimulus.lower():
                    chk_num = idx + 1  # MATLAB indices start at 1
                    break
            if chk_num is None:
                raise ValueError('Invalid stimulus for the given session.')

        # Construct the participant ID
        if not stimulus:
            participant_id = f"OE1{path_num}{subject_id_str}"
        else:
            participant_id = f"OE1{path_num}{subject_id_str}_{chk_num}"
        return participant_id

    elif city.lower() == 'copenhagen':
        # Placeholder for Copenhagen logic
        raise NotImplementedError('Logic for Copenhagen is not implemented yet.')
    else:
        raise ValueError('City not supported.')


def parse_participant_code(participant_code):
    """
    Parses the participant code to extract city, subject ID, session, and stimulus.

    Parameters:
    - participant_code (str): The participant code (e.g., 'OE106005' or 'OE104003_1').

    Returns:
    - dict: A dictionary containing 'city', 'subject_id', 'session', and 'stimulus'.

    Raises:
    - ValueError: If the participant code is invalid.

    Example:
    --------
    participant_info = parse_participant_code('OE104003_1')
    print(participant_info)
    # Output: {'city': 'Lisbon', 'subject_id': 3, 'session': 'Baixa', 'stimulus': 'D12'}
    """

    # Initialize mappings (same as in infer_participant_code)
    sessions = [
        ('Baixa', 4),
        ('Belem', 1),
        ('Parque', 6),
        ('Gulbenkian', 3),
        ('Lapa', 2),
        ('Graca', 5)
    ]

    # Reverse mapping for session numbers to names
    session_num_to_name = {number: name for name, number in sessions}

    checkpoints = [
        ['A7', 'C9', 'C10'],            # Belem - 1
        ['A16', 'C17', 'C18'],          # Lapa - 2
        ['A3', 'B4', 'B6', 'B7'],       # Gulbenkian - 3
        ['D12', 'D27', 'D26', 'B10'],   # Baixa - 4
        ['C31', 'C32', 'D31', 'D32'],   # Graca - 5
        ['A24', 'A25']                  # Parque das Nações - 6
    ]

    if participant_code.startswith('OE1'):
        # Extract path_num and subject_id
        rest = participant_code[3:]  # Remove 'OE1'
        if '_' in rest:
            # Participant code includes stimulus
            main_part, chk_num_str = rest.split('_')
            chk_num = int(chk_num_str)
        else:
            main_part = rest
            chk_num = None

        if len(main_part) < 5:
            raise ValueError('Invalid participant code format.')

        path_num_str = main_part[:2]
        subject_id_str = main_part[2:]

        # Convert to integers
        try:
            ses_num = int(path_num_str)
            subject_id = int(subject_id_str)
        except ValueError:
            raise ValueError('Invalid participant code format.')

        # Get session name
        session = session_num_to_name.get(ses_num)
        if not session:
            raise ValueError('Invalid session number in participant code.')

        # Get stimulus if available
        if chk_num is not None:
            try:
                stimulus = checkpoints[ses_num - 1][chk_num - 1]
            except IndexError:
                raise ValueError('Invalid checkpoint number in participant code.')
        else:
            stimulus = ''

        participant_info = {
            'city': 'Lisbon',
            'subject_id': subject_id,
            'session': session,
            'stimulus': stimulus
        }
        return participant_info
    else:
        raise ValueError('Participant code format not recognized or city not supported.')



