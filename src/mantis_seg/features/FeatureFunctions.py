# Copyright (C) 2023 Friedrich Miescher Institute for Biomedical Research

##############################################################################
#                                                                            #
# Author: Nicole Repina              <nicole.repina@fmi.ch>                  #
#                                                                            #
##############################################################################

import math


def mesh_equivalent_diameter(volume):
    """
    Calculate equivalent diameter of sphere with a given volume
    """
    try:
        equiv_diam = 2 * ((0.75 * volume * (1 / math.pi)) ** (1.0 / 3.0))
    except ValueError:
        equiv_diam = 0
    return equiv_diam