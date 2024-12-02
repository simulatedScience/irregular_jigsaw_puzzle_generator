"""
This module provides functionality to procedurally generate interlocking connectors for jigsaw puzzle pieces.
Each connector is meant to be placed on one polygon edge to connect the pieces sharing that edge.
"""

class jigsaw_connector:
    """
    When a new instance of this class is created, a new connector is randomly generated. 
    """
    def __init__(self):
        