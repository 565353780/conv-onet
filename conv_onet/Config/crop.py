#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

UNIT_SPACE_EXPAND = 0.01

UNIT_LB = np.array([-0.5, -0.5, -0.5]) - UNIT_SPACE_EXPAND
UNIT_UB = np.array([0.5, 0.5, 0.5]) + UNIT_SPACE_EXPAND
