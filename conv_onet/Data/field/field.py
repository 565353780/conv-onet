#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Field(object):
    ''' Data fields class.
    '''

    def load(self, data_path, idx, category):
        ''' Loads a data point.

        Args:
            data_path (str): path to data file
            idx (int): index of data point
            category (int): index of category
        '''
        raise NotImplementedError

    def check_complete(self, filed):
        ''' Checks if set is complete.

        Args:
            files: files
        '''
        raise NotImplementedError
