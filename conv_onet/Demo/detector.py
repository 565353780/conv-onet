#!/usr/bin/env python
# -*- coding: utf-8 -*-

from conv_onet.Module.detector import Detector


def demo():
    detector = Detector()
    detector.detectAll()
    return True
