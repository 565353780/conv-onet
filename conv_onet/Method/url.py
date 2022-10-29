#!/usr/bin/env python
# -*- coding: utf-8 -*-

import urllib


def is_url(url):
    scheme = urllib.parse.urlparse(url).scheme
    return scheme in ('http', 'https')
