def ordinal(number):
    """
    Convert an integer into its ordinal representation:
    1 -> 1st, 2 -> 2nd, 3 -> 3rd, 4 -> 4th, etc.
    """
    if 10 <= number % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
    return f"{number}{suffix}"
